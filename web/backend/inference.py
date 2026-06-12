"""
Inference pipeline for cattle identification.
Wraps the GNN pipeline with SIFT fallback when no trained model is available.

Pipeline:
  Image → Preprocessing (ROI + CLAHE) → SuperPoint/SIFT Descriptors
       → KNN Graph → CattleGNN Forward → 256-d L2 Embedding
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add parent src to path so we can import the GNN modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Configuration defaults
# ────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pt"))
MAX_KEYPOINTS = int(os.getenv("MAX_KEYPOINTS", "128"))
IMAGE_SIZE = 256
DESCRIPTOR_DIM = 256


# ────────────────────────────────────────────────────────────────────────────
# Image Preprocessing helpers
# ────────────────────────────────────────────────────────────────────────────
def preprocess_image(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply ROI extraction, CLAHE enhancement and resize.
    Returns (processed_bgr, gray).
    """
    # Resize to standard size
    img = cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    return img, gray_enhanced


# ────────────────────────────────────────────────────────────────────────────
# SIFT Fallback Extractor
# ────────────────────────────────────────────────────────────────────────────
class SIFTExtractor:
    """Fallback feature extractor using SIFT, pads descriptors to 256-d."""

    def __init__(self, max_keypoints: int = 128):
        self.max_keypoints = max_keypoints
        self.sift = cv2.SIFT_create(nfeatures=max_keypoints)
        logger.info("Using SIFT fallback extractor (no trained SuperPoint model)")

    def extract(self, gray: np.ndarray) -> Dict[str, np.ndarray]:
        kps, descs = self.sift.detectAndCompute(gray, None)
        if kps is None or len(kps) == 0:
            return {
                "keypoints": np.zeros((0, 2), dtype=np.float32),
                "descriptors": np.zeros((0, DESCRIPTOR_DIM), dtype=np.float32),
                "scores": np.zeros(0, dtype=np.float32),
            }

        keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
        scores = np.array([kp.response for kp in kps], dtype=np.float32)

        # SIFT gives 128-d; pad to 256-d
        if descs.shape[1] < DESCRIPTOR_DIM:
            pad = np.zeros((descs.shape[0], DESCRIPTOR_DIM - descs.shape[1]), dtype=np.float32)
            descs = np.hstack([descs, pad])

        return {"keypoints": keypoints, "descriptors": descs.astype(np.float32), "scores": scores}


# ────────────────────────────────────────────────────────────────────────────
# SuperPoint Extractor (using kornia)
# ────────────────────────────────────────────────────────────────────────────
class SuperPointExtractor:
    def __init__(self, max_keypoints: int = 128, device: str = "cpu"):
        self.max_keypoints = max_keypoints
        self.device = device
        self.model = self._load()

    def _load(self):
        try:
            from kornia.feature import SuperPoint as KorniaSP
            model = KorniaSP(
                num_features=self.max_keypoints,
                detection_threshold=0.005,
                nms_radius=4,
            ).to(self.device)
            model.eval()
            logger.info(f"SuperPoint loaded on {self.device}")
            return model
        except Exception as e:
            logger.warning(f"SuperPoint unavailable: {e}. Using SIFT.")
            return None

    def extract(self, gray: np.ndarray) -> Dict[str, np.ndarray]:
        if self.model is None:
            return SIFTExtractor(self.max_keypoints).extract(gray)

        img_t = torch.from_numpy(gray).float() / 255.0
        img_t = img_t.unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(img_t)

        kps = out["keypoints"][0].cpu().numpy()
        scores = out["keypoint_scores"][0].cpu().numpy()
        descs = out["descriptors"][0].cpu().numpy()

        # Keep top-k
        if len(kps) > self.max_keypoints:
            idx = np.argsort(scores)[::-1][: self.max_keypoints]
            kps, scores, descs = kps[idx], scores[idx], descs[idx]

        return {"keypoints": kps, "descriptors": descs, "scores": scores}


# ────────────────────────────────────────────────────────────────────────────
# Simple mean-pooling embedding (no GNN, just average descriptor)
# ────────────────────────────────────────────────────────────────────────────
def descriptors_to_embedding(descriptors: np.ndarray) -> np.ndarray:
    """Pool descriptors into a single 256-d L2-normalised embedding."""
    if len(descriptors) == 0:
        return np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
    emb = descriptors.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


# ────────────────────────────────────────────────────────────────────────────
# GNN-based embedding (when trained model available)
# ────────────────────────────────────────────────────────────────────────────
def try_load_gnn_model(model_path: str, device: str):
    """Try to load the trained CattleGNN checkpoint."""
    try:
        from src.models.gnn_model import CattleGNN
        from src.features.graph_builder import GraphBuilder
        import yaml

        config_path = PROJECT_ROOT / "config" / "config.yaml"
        with open(config_path) as f:
            import yaml
            cfg = yaml.safe_load(f)

        model = CattleGNN(config=cfg)
        state = torch.load(model_path, map_location=device)
        # Handle common checkpoint formats
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        model.to(device)
        logger.info(f"CattleGNN loaded from {model_path}")
        return model, cfg
    except Exception as e:
        logger.warning(f"Could not load GNN model: {e}. Will use descriptor averaging.")
        return None, None


def gnn_embedding(
    gray: np.ndarray,
    descriptors: np.ndarray,
    keypoints: np.ndarray,
    model,
    cfg: dict,
    device: str,
) -> np.ndarray:
    """Build graph and run GNN forward pass to get 256-d embedding."""
    try:
        from src.features.graph_builder import GraphBuilder
        builder = GraphBuilder(config=cfg)
        data = builder.build(
            descriptors=descriptors,
            keypoints=keypoints,
            image_size=(gray.shape[1], gray.shape[0]),
        )
        data = data.to(device)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(data)
        emb = out["embedding"].squeeze(0).cpu().numpy()
        return emb.astype(np.float32)
    except Exception as e:
        logger.warning(f"GNN forward failed: {e}. Falling back to descriptor avg.")
        return descriptors_to_embedding(descriptors)


# ────────────────────────────────────────────────────────────────────────────
# Main Inference Engine (singleton per process)
# ────────────────────────────────────────────────────────────────────────────
class CattleInferenceEngine:
    _instance = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Inference engine using device: {self.device}")

        # Feature extractor (SuperPoint → SIFT fallback)
        self.extractor = SuperPointExtractor(max_keypoints=MAX_KEYPOINTS, device=self.device)

        # GNN model (optional)
        self.gnn_model, self.gnn_cfg = None, None
        if Path(MODEL_PATH).exists():
            self.gnn_model, self.gnn_cfg = try_load_gnn_model(MODEL_PATH, self.device)

        if self.gnn_model is None:
            logger.info("Running in descriptor-averaging mode (no trained model)")

    @classmethod
    def get(cls) -> "CattleInferenceEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_image_bytes(self, image_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        End-to-end: raw image bytes → 256-d embedding.
        Returns (embedding, info_dict).
        """
        # Decode image
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image. Ensure it is a valid JPEG/PNG.")

        # Preprocess
        img_proc, gray = preprocess_image(img)

        # Extract features
        feat = self.extractor.extract(gray)
        kps = feat["keypoints"]
        descs = feat["descriptors"]
        scores = feat["scores"]

        extractor_name = "superpoint" if (
            self.extractor.model is not None
        ) else "sift"

        if len(descs) == 0:
            # Very edge case – blank image
            embedding = np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
        elif self.gnn_model is not None:
            embedding = gnn_embedding(gray, descs, kps, self.gnn_model, self.gnn_cfg, self.device)
            extractor_name = "superpoint+gnn"
        else:
            embedding = descriptors_to_embedding(descs)

        info = {
            "num_keypoints": len(kps),
            "confidence": float(np.mean(scores)) if len(scores) > 0 else 0.0,
            "extractor": extractor_name,
            "model_version": "CattleGNN-v1" if self.gnn_model else "descriptor-avg-v1",
        }

        return embedding, info
