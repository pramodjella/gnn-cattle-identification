"""
FastAPI application – Cattle Biometric Identification Platform
Endpoints:
  POST /api/cattle/register   – Register new cattle with muzzle photo
  POST /api/cattle/identify   – Identify cattle from muzzle photo
  GET  /api/cattle/           – List all cattle
  GET  /api/cattle/{id}       – Get cattle details
  PUT  /api/cattle/{id}       – Update cattle metadata
  DELETE /api/cattle/{id}     – Remove cattle + embeddings
  GET  /api/stats             – System statistics
  GET  /api/logs              – Identification logs
"""

import os
import uuid
import shutil
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Any
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text, select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db, Cattle, CattleEmbedding, IdentificationLog, engine, Base
from inference import CattleInferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
TOP_K = int(os.getenv("TOP_K", "5"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure tables exist
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
        await conn.run_sync(Base.metadata.create_all)
    # Pre-load inference engine
    engine_inst = CattleInferenceEngine.get()
    logger.info("Cattle Identification Platform ready")
    yield


app = FastAPI(
    title="Cattle Biometric Identification API",
    description="GNN-powered cattle muzzle biometric registration & identification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


# ── Pydantic Schemas ────────────────────────────────────────────────────────
class CattleOut(BaseModel):
    id: str
    tag_id: str
    name: Optional[str]
    breed: Optional[str]
    sex: Optional[str]
    date_of_birth: Optional[date]
    farm_name: Optional[str]
    farm_location: Optional[str]
    owner_name: Optional[str]
    owner_contact: Optional[str]
    weight_kg: Optional[float]
    notes: Optional[str]
    photo_url: Optional[str]
    status: str
    embedding_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class MatchResult(BaseModel):
    rank: int
    cattle_id: str
    tag_id: str
    name: Optional[str]
    breed: Optional[str]
    farm_name: Optional[str]
    similarity: float
    photo_url: Optional[str]
    accepted: bool


class IdentifyResponse(BaseModel):
    query_id: str
    accepted: bool
    top_match: Optional[MatchResult]
    top_k: List[MatchResult]
    threshold: float
    extractor: str
    model_version: str
    num_keypoints: int


class StatsOut(BaseModel):
    active_cattle: int
    total_cattle: int
    total_embeddings: int
    total_identifications: int
    successful_identifications: int


# ── Helpers ─────────────────────────────────────────────────────────────────
def photo_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"/uploads/{Path(path).name}"


async def cattle_to_out(c: Cattle, db: AsyncSession) -> CattleOut:
    count_result = await db.execute(
        select(func.count()).where(CattleEmbedding.cattle_id == c.id)
    )
    count = count_result.scalar() or 0
    return CattleOut(
        id=str(c.id),
        tag_id=c.tag_id,
        name=c.name,
        breed=c.breed,
        sex=c.sex,
        date_of_birth=c.date_of_birth,
        farm_name=c.farm_name,
        farm_location=c.farm_location,
        owner_name=c.owner_name,
        owner_contact=c.owner_contact,
        weight_kg=c.weight_kg,
        notes=c.notes,
        photo_url=photo_url(c.photo_path),
        status=c.status,
        embedding_count=count,
        created_at=c.created_at,
    )


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/stats", response_model=StatsOut)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """System-wide statistics."""
    result = await db.execute(text("SELECT * FROM system_stats"))
    row = result.fetchone()
    if row:
        return StatsOut(
            active_cattle=row.active_cattle,
            total_cattle=row.total_cattle,
            total_embeddings=row.total_embeddings,
            total_identifications=row.total_identifications,
            successful_identifications=row.successful_identifications,
        )
    return StatsOut(active_cattle=0, total_cattle=0, total_embeddings=0,
                    total_identifications=0, successful_identifications=0)


@app.post("/api/cattle/register", response_model=CattleOut)
async def register_cattle(
    tag_id: str = Form(...),
    name: Optional[str] = Form(None),
    breed: Optional[str] = Form(None),
    sex: Optional[str] = Form(None),
    date_of_birth: Optional[str] = Form(None),
    farm_name: Optional[str] = Form(None),
    farm_location: Optional[str] = Form(None),
    owner_name: Optional[str] = Form(None),
    owner_contact: Optional[str] = Form(None),
    weight_kg: Optional[float] = Form(None),
    notes: Optional[str] = Form(None),
    muzzle_image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Register a new cattle with muzzle photo. Extracts and stores GNN embedding."""

    # Check duplicate tag
    existing = await db.execute(select(Cattle).where(Cattle.tag_id == tag_id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Cattle with tag_id '{tag_id}' already exists.")

    # Save image
    ext = Path(muzzle_image.filename or "image.jpg").suffix or ".jpg"
    img_filename = f"{uuid.uuid4()}{ext}"
    img_path = UPLOAD_DIR / img_filename
    img_bytes = await muzzle_image.read()
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    # Extract embedding
    try:
        inf = CattleInferenceEngine.get()
        embedding, info = inf.embed_image_bytes(img_bytes)
    except Exception as e:
        img_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Feature extraction failed: {e}")

    # Parse date
    dob = None
    if date_of_birth:
        try:
            dob = date.fromisoformat(date_of_birth)
        except ValueError:
            pass

    # Create cattle record
    cattle = Cattle(
        tag_id=tag_id,
        name=name,
        breed=breed,
        sex=sex,
        date_of_birth=dob,
        farm_name=farm_name,
        farm_location=farm_location,
        owner_name=owner_name,
        owner_contact=owner_contact,
        weight_kg=weight_kg,
        notes=notes,
        photo_path=str(img_path),
    )
    db.add(cattle)
    await db.flush()  # get the UUID

    # Store embedding
    emb_record = CattleEmbedding(
        cattle_id=cattle.id,
        embedding=embedding.tolist(),
        image_path=str(img_path),
        model_version=info["model_version"],
        extractor=info["extractor"],
        num_keypoints=info["num_keypoints"],
        confidence=info["confidence"],
    )
    db.add(emb_record)
    await db.commit()
    await db.refresh(cattle)

    logger.info(f"Registered cattle {tag_id} with {info['num_keypoints']} keypoints")
    return await cattle_to_out(cattle, db)


@app.post("/api/cattle/identify", response_model=IdentifyResponse)
async def identify_cattle(
    muzzle_image: UploadFile = File(...),
    top_k: int = Query(default=5, le=20),
    threshold: Optional[float] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Identify cattle from a muzzle image. Returns top-k matches with similarity scores."""

    sim_threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD
    img_bytes = await muzzle_image.read()

    # Save query image
    ext = Path(muzzle_image.filename or "query.jpg").suffix or ".jpg"
    query_filename = f"query_{uuid.uuid4()}{ext}"
    query_path = UPLOAD_DIR / query_filename
    with open(query_path, "wb") as f:
        f.write(img_bytes)

    # Extract embedding
    try:
        inf = CattleInferenceEngine.get()
        embedding, info = inf.embed_image_bytes(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature extraction failed: {e}")

    emb_list = embedding.tolist()

    # pgvector cosine similarity search
    # cosine_distance = 1 - cosine_similarity; we want 1 - distance
    sql = text("""
        SELECT
            ce.cattle_id::text,
            1 - (ce.embedding <=> CAST(:emb AS vector)) AS similarity
        FROM cattle_embeddings ce
        JOIN cattle c ON c.id = ce.cattle_id
        WHERE c.status = 'active'
        ORDER BY ce.embedding <=> CAST(:emb AS vector)
        LIMIT :k
    """)
    result = await db.execute(sql, {"emb": str(emb_list), "k": top_k * 3})
    rows = result.fetchall()

    # Aggregate: for each cattle_id, take max similarity
    best: dict[str, float] = {}
    for row in rows:
        cid = row.cattle_id
        sim = float(row.similarity)
        if cid not in best or sim > best[cid]:
            best[cid] = sim

    # Sort and take top-k unique cattle
    top_cattle = sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]

    matches: List[MatchResult] = []
    for rank, (cid, sim) in enumerate(top_cattle, 1):
        cattle_row = await db.execute(select(Cattle).where(Cattle.id == cid))
        c = cattle_row.scalar_one_or_none()
        if c:
            matches.append(MatchResult(
                rank=rank,
                cattle_id=cid,
                tag_id=c.tag_id,
                name=c.name,
                breed=c.breed,
                farm_name=c.farm_name,
                similarity=round(sim, 4),
                photo_url=photo_url(c.photo_path),
                accepted=sim >= sim_threshold,
            ))

    top_match = matches[0] if matches else None
    accepted = bool(top_match and top_match.similarity >= sim_threshold)

    # Log
    log = IdentificationLog(
        query_image=str(query_path),
        matched_cattle_id=uuid.UUID(top_match.cattle_id) if top_match else None,
        similarity=top_match.similarity if top_match else None,
        threshold=sim_threshold,
        accepted=accepted,
        top_k_results=[m.dict() for m in matches],
        extractor=info["extractor"],
        model_version=info["model_version"],
    )
    db.add(log)
    await db.commit()

    return IdentifyResponse(
        query_id=str(log.id),
        accepted=accepted,
        top_match=top_match,
        top_k=matches,
        threshold=sim_threshold,
        extractor=info["extractor"],
        model_version=info["model_version"],
        num_keypoints=info["num_keypoints"],
    )


@app.get("/api/cattle/", response_model=List[CattleOut])
async def list_cattle(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, le=200),
    status: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """List all registered cattle with optional filters."""
    q = select(Cattle).order_by(desc(Cattle.created_at)).offset(skip).limit(limit)
    if status:
        q = q.where(Cattle.status == status)
    if search:
        pattern = f"%{search}%"
        from sqlalchemy import or_
        q = q.where(or_(
            Cattle.tag_id.ilike(pattern),
            Cattle.name.ilike(pattern),
            Cattle.breed.ilike(pattern),
            Cattle.farm_name.ilike(pattern),
        ))
    result = await db.execute(q)
    cattle_list = result.scalars().all()
    return [await cattle_to_out(c, db) for c in cattle_list]


@app.get("/api/cattle/{cattle_id}", response_model=CattleOut)
async def get_cattle(cattle_id: str, db: AsyncSession = Depends(get_db)):
    """Get details for a specific cattle."""
    result = await db.execute(select(Cattle).where(Cattle.id == cattle_id))
    c = result.scalar_one_or_none()
    if not c:
        raise HTTPException(status_code=404, detail="Cattle not found")
    return await cattle_to_out(c, db)


@app.put("/api/cattle/{cattle_id}", response_model=CattleOut)
async def update_cattle(
    cattle_id: str,
    name: Optional[str] = Form(None),
    breed: Optional[str] = Form(None),
    sex: Optional[str] = Form(None),
    date_of_birth: Optional[str] = Form(None),
    farm_name: Optional[str] = Form(None),
    farm_location: Optional[str] = Form(None),
    owner_name: Optional[str] = Form(None),
    owner_contact: Optional[str] = Form(None),
    weight_kg: Optional[float] = Form(None),
    notes: Optional[str] = Form(None),
    status: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Update cattle metadata."""
    result = await db.execute(select(Cattle).where(Cattle.id == cattle_id))
    c = result.scalar_one_or_none()
    if not c:
        raise HTTPException(status_code=404, detail="Cattle not found")

    for field, value in {
        "name": name, "breed": breed, "sex": sex,
        "farm_name": farm_name, "farm_location": farm_location,
        "owner_name": owner_name, "owner_contact": owner_contact,
        "weight_kg": weight_kg, "notes": notes, "status": status,
    }.items():
        if value is not None:
            setattr(c, field, value)

    if date_of_birth:
        try:
            c.date_of_birth = date.fromisoformat(date_of_birth)
        except ValueError:
            pass

    await db.commit()
    await db.refresh(c)
    return await cattle_to_out(c, db)


@app.delete("/api/cattle/{cattle_id}")
async def delete_cattle(cattle_id: str, db: AsyncSession = Depends(get_db)):
    """Remove cattle and all embeddings."""
    result = await db.execute(select(Cattle).where(Cattle.id == cattle_id))
    c = result.scalar_one_or_none()
    if not c:
        raise HTTPException(status_code=404, detail="Cattle not found")
    await db.delete(c)
    await db.commit()
    return {"message": f"Cattle {cattle_id} deleted"}


@app.get("/api/logs")
async def get_logs(
    skip: int = Query(default=0),
    limit: int = Query(default=50, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Identification attempt logs."""
    result = await db.execute(
        select(IdentificationLog).order_by(desc(IdentificationLog.created_at)).offset(skip).limit(limit)
    )
    logs = result.scalars().all()
    return [
        {
            "id": str(log.id),
            "matched_cattle_id": str(log.matched_cattle_id) if log.matched_cattle_id else None,
            "similarity": log.similarity,
            "threshold": log.threshold,
            "accepted": log.accepted,
            "extractor": log.extractor,
            "model_version": log.model_version,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in logs
    ]


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
