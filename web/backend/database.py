"""
Database configuration, models and session management.
Uses SQLAlchemy async with pgvector for embedding storage.
"""

import os
from datetime import date, datetime
from typing import Optional, List
from uuid import UUID, uuid4

from sqlalchemy import (
    String, Text, Float, Integer, Boolean, Date,
    DateTime, Enum, ForeignKey, func, Column
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://cattle_user:cattle_pass@localhost:5432/cattle_db"
)

# ── Engine & Session ────────────────────────────────────────────────────────
engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# ── ORM Models ──────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class Cattle(Base):
    __tablename__ = "cattle"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tag_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(128))
    breed: Mapped[Optional[str]] = mapped_column(String(64))
    sex: Mapped[Optional[str]] = mapped_column(String(16))
    date_of_birth: Mapped[Optional[date]] = mapped_column(Date)
    farm_name: Mapped[Optional[str]] = mapped_column(String(128))
    farm_location: Mapped[Optional[str]] = mapped_column(String(256))
    owner_name: Mapped[Optional[str]] = mapped_column(String(128))
    owner_contact: Mapped[Optional[str]] = mapped_column(String(64))
    weight_kg: Mapped[Optional[float]] = mapped_column(Float)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    photo_path: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    embeddings: Mapped[List["CattleEmbedding"]] = relationship(
        "CattleEmbedding", back_populates="cattle", cascade="all, delete-orphan"
    )


class CattleEmbedding(Base):
    __tablename__ = "cattle_embeddings"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    cattle_id: Mapped[UUID] = mapped_column(ForeignKey("cattle.id", ondelete="CASCADE"), nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Vector(256), nullable=False)
    image_path: Mapped[Optional[str]] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(String(64), default="CattleGNN-v1")
    extractor: Mapped[str] = mapped_column(String(32), default="superpoint")
    num_keypoints: Mapped[Optional[int]] = mapped_column(Integer)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    cattle: Mapped["Cattle"] = relationship("Cattle", back_populates="embeddings")


class IdentificationLog(Base):
    __tablename__ = "identification_logs"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    query_image: Mapped[Optional[str]] = mapped_column(Text)
    matched_cattle_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("cattle.id"), nullable=True)
    similarity: Mapped[Optional[float]] = mapped_column(Float)
    threshold: Mapped[Optional[float]] = mapped_column(Float)
    accepted: Mapped[Optional[bool]] = mapped_column(Boolean)
    top_k_results: Mapped[Optional[dict]] = mapped_column(JSONB)
    extractor: Mapped[Optional[str]] = mapped_column(String(32))
    model_version: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
