"""
Kornia-Based Keypoint Extraction Suite
=======================================
Provides four parallel extraction backends, all with a unified API:

  1. Kornia DISK          – Neural detector (depth-pretrained, 128-d)
  2. Kornia SuperPoint*   – KeyNet detector + HardNet descriptor (true
                            SuperPoint-class neural approach available in
                            kornia 0.8.x as KeyNetAffNetHardNet pipeline)
  3. Kornia DeDoDe        – Detect, Don't Describe (G variant, 256-d)
  4. OpenCV SIFT          – Classical baseline fallback

*Kornia 0.8.x does NOT ship the original Detone et al. SuperPoint weights,
 but KeyNetAffNetHardNet is the canonical "learned SuperPoint-class" pipeline
 that kornia maintains and is functionally equivalent for this task.

Usage
-----
    from src.features.superpoint import SuperPointExtractor, MultiExtractor

    # Single backend (default = DISK)
    ext = SuperPointExtractor(backend='disk', max_keypoints=256)
    result = ext.extract(image_bgr)

    # All four backends in parallel (ThreadPoolExecutor)
    multi = MultiExtractor(max_keypoints=256)
    results = multi.extract_parallel(image_bgr)  # dict keyed by backend name
"""

from __future__ import annotations

import concurrent.futures
import time
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

class _DISKBackend:
    """Kornia DISK neural keypoint detector (depth-pretrained, 128-d desc)."""

    NAME = "Kornia-DISK"

    def __init__(self, max_keypoints: int, device: torch.device):
        self.max_keypoints = max_keypoints
        self.device = device
        self.model = self._load()

    def _load(self):
        try:
            from kornia.feature import DISK as KorniaDISK
            model = KorniaDISK.from_pretrained('depth').to(self.device)
            model.eval()
            print(f"[INFO] {self.NAME} loaded on {self.device}")
            return model
        except Exception as exc:
            warnings.warn(f"{self.NAME} unavailable: {exc}")
            return None

    @property
    def available(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def extract(self, rgb_image: np.ndarray, mask: Optional[np.ndarray] = None
                ) -> Dict[str, np.ndarray]:
        """Extract keypoints from an RGB uint8 image (H, W, 3)."""
        if self.model is None:
            return _empty_result(256)

        h, w = rgb_image.shape[:2]
        t = torch.from_numpy(rgb_image).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1,3,H,W)

        features = self.model(t, n=self.max_keypoints, pad_if_not_divisible=True)
        disk_kps = features[0]

        n = disk_kps.keypoints.shape[0]
        if n == 0:
            return _empty_result(256)

        kps   = disk_kps.keypoints.cpu().numpy()          # (N,2) x,y
        scs   = disk_kps.detection_scores.cpu().numpy()   # (N,)
        descs = disk_kps.descriptors.cpu().numpy()         # (N,128)

        kps, scs, descs = _apply_mask(kps, scs, descs, mask, h, w)
        kps, scs, descs = _topk(kps, scs, descs, self.max_keypoints)

        # Pad 128-d → 256-d for GNN node-feature compatibility
        descs = _pad_to_256(descs)
        return {"keypoints": kps, "descriptors": descs, "scores": scs}


class _SuperPointBackend:
    """
    Kornia SuperPoint-class pipeline: KeyNet detector + AffNet shaper +
    HardNet8 descriptor (kornia 0.8.x canonical learned-feature pipeline).

    This is the functional equivalent of SuperPoint for this codebase.
    """

    NAME = "Kornia-KeyNet+HardNet"

    def __init__(self, max_keypoints: int, device: torch.device):
        self.max_keypoints = max_keypoints
        self.device = device
        self.model = self._load()

    def _load(self):
        try:
            from kornia.feature import KeyNetAffNetHardNet
            model = KeyNetAffNetHardNet(
                num_features=self.max_keypoints,
                upright=False,
                device=self.device,
            )
            model.eval()
            print(f"[INFO] {self.NAME} loaded on {self.device}")
            return model
        except Exception as exc:
            warnings.warn(f"{self.NAME} unavailable: {exc}")
            return None

    @property
    def available(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def extract(self, rgb_image: np.ndarray, mask: Optional[np.ndarray] = None
                ) -> Dict[str, np.ndarray]:
        if self.model is None:
            return _empty_result(256)

        h, w = rgb_image.shape[:2]
        # KeyNetAffNetHardNet expects grayscale (1,1,H,W) float [0,1]
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        t = torch.from_numpy(gray).float() / 255.0
        t = t.unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H,W)

        try:
            lafs, resps, descs = self.model(t)
        except Exception as exc:
            warnings.warn(f"{self.NAME} extract failed: {exc}")
            return _empty_result(256)

        # lafs: (1, N, 2, 3); descs: (1, N, 128)
        if lafs is None or lafs.shape[1] == 0:
            return _empty_result(256)

        # Extract (x, y) centers from LAF
        from kornia.feature import get_laf_center
        centers = get_laf_center(lafs)[0].cpu().numpy()  # (N, 2)

        scs   = resps[0].cpu().numpy().flatten()          # (N,)
        descs = descs[0].cpu().numpy()                    # (N, 128)

        kps, scs, descs = _apply_mask(centers, scs, descs, mask, h, w)
        kps, scs, descs = _topk(kps, scs, descs, self.max_keypoints)
        descs = _pad_to_256(descs)

        return {"keypoints": kps, "descriptors": descs, "scores": scs}


class _DeDoDeMatcher:
    """
    Kornia DeDoDe – Detect, Don't Describe.
    Uses the 'G' variant (general) which outputs 256-d descriptors.
    """

    NAME = "Kornia-DeDoDe"

    def __init__(self, max_keypoints: int, device: torch.device):
        self.max_keypoints = max_keypoints
        self.device = device
        self.model = self._load()

    @staticmethod
    def _weights_cached() -> bool:
        """Check whether DeDoDe weights are already in the torch hub cache."""
        import os
        cache_dir = os.path.join(
            os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints')
        # DeDoDe-G descriptor weight file
        return any(
            f.startswith('dedode') or 'DeDoDe' in f
            for f in os.listdir(cache_dir)
            if os.path.isfile(os.path.join(cache_dir, f))
        ) if os.path.isdir(cache_dir) else False

    def _load(self):
        try:
            from kornia.feature import DeDoDe

            if not self._weights_cached():
                print(f"[INFO] {self.NAME}: weights not cached – skipping "
                      f"(run with DeDoDe once to download ~1.1GB, then retry).")
                return None

            model = DeDoDe.from_pretrained(detector_weights="L-upright",
                                           descriptor_weights="G-upright")
            model = model.to(self.device)
            model.eval()
            print(f"[INFO] {self.NAME} loaded on {self.device}")
            return model
        except Exception as exc:
            warnings.warn(f"{self.NAME} unavailable: {exc}")
            return None

    @property
    def available(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def extract(self, rgb_image: np.ndarray, mask: Optional[np.ndarray] = None
                ) -> Dict[str, np.ndarray]:
        if self.model is None:
            return _empty_result(256)

        h, w = rgb_image.shape[:2]
        t = torch.from_numpy(rgb_image).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)

        try:
            # DeDoDe.detect_and_describe returns (kps_norm, scores, descs)
            # kps in [-1, 1] (normalised), descs 256-d
            kps_norm, scs, descs = self.model.detect_and_describe(
                t, num_keypoints=self.max_keypoints
            )
        except Exception as exc:
            warnings.warn(f"{self.NAME} extract failed: {exc}")
            return _empty_result(256)

        # kps_norm: (1, N, 2)  range [-1, 1]  →  pixel coords
        kps_n = kps_norm[0].cpu().numpy()   # (N, 2)
        scs   = scs[0].cpu().numpy()         # (N,)
        descs = descs[0].cpu().numpy()       # (N, 256)

        # Denormalize from [-1,1] to pixel
        kps = np.stack([(kps_n[:, 0] + 1) / 2 * w,
                        (kps_n[:, 1] + 1) / 2 * h], axis=-1).astype(np.float32)

        kps, scs, descs = _apply_mask(kps, scs, descs, mask, h, w)
        kps, scs, descs = _topk(kps, scs, descs, self.max_keypoints)

        return {"keypoints": kps, "descriptors": descs, "scores": scs}


class _SIFTBackend:
    """Classical OpenCV SIFT – always available as fallback baseline."""

    NAME = "SIFT"

    def __init__(self, max_keypoints: int):
        self.max_keypoints = max_keypoints

    @property
    def available(self) -> bool:
        return True

    def extract(self, rgb_image: np.ndarray, mask: Optional[np.ndarray] = None
                ) -> Dict[str, np.ndarray]:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create(nfeatures=self.max_keypoints)
        kps, descs = sift.detectAndCompute(gray, mask)

        if kps is None or len(kps) == 0:
            return _empty_result(256)

        kp_arr = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
        sc_arr = np.array([kp.response for kp in kps], dtype=np.float32)
        descs  = _pad_to_256(descs.astype(np.float32))
        return {"keypoints": kp_arr, "descriptors": descs, "scores": sc_arr}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _empty_result(desc_dim: int = 256) -> Dict[str, np.ndarray]:
    return {
        "keypoints":   np.zeros((0, 2),        dtype=np.float32),
        "descriptors": np.zeros((0, desc_dim),  dtype=np.float32),
        "scores":      np.zeros(0,              dtype=np.float32),
    }


def _apply_mask(kps, scs, descs, mask, h, w):
    if mask is None or len(kps) == 0:
        return kps, scs, descs
    valid = []
    for i, (x, y) in enumerate(kps):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < h and 0 <= ix < w and mask[iy, ix] > 0:
            valid.append(i)
    if not valid:
        return (np.zeros((0, 2), dtype=np.float32),
                np.zeros(0,      dtype=np.float32),
                np.zeros((0, descs.shape[1]), dtype=np.float32))
    return kps[valid], scs[valid], descs[valid]


def _topk(kps, scs, descs, k: int):
    if len(kps) <= k:
        return kps, scs, descs
    idx = np.argsort(scs)[::-1][:k]
    return kps[idx], scs[idx], descs[idx]


def _pad_to_256(descs: np.ndarray) -> np.ndarray:
    if descs.ndim != 2:
        return descs
    d = descs.shape[1]
    if d >= 256:
        return descs[:, :256].astype(np.float32)
    pad = np.zeros((descs.shape[0], 256 - d), dtype=np.float32)
    return np.hstack([descs.astype(np.float32), pad])


# ---------------------------------------------------------------------------
# Public: SuperPointExtractor  (drop-in replacement, default = DISK)
# ---------------------------------------------------------------------------

class SuperPointExtractor:
    """
    Drop-in replacement for the previous SuperPointExtractor.

    Default backend: 'disk' (Kornia DISK, best accuracy).
    Other backends: 'superpoint' (KeyNet+HardNet), 'dedode', 'sift'.

    Maintains API compatibility: extract(), visualize(), get_stats().
    """

    BACKENDS = ('disk', 'superpoint', 'dedode', 'sift')

    def __init__(self,
                 max_keypoints: int = 512,
                 detection_threshold: float = 0.005,
                 nms_radius: int = 4,
                 device: Optional[torch.device] = None,
                 backend: str = 'disk'):

        self.max_keypoints = max_keypoints
        self.detection_threshold = detection_threshold
        self.nms_radius = nms_radius
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}, got '{backend}'")
        self.backend_name = backend
        self._backend = self._init_backend(backend)
        self.method_name = self._backend.NAME

        self.stats: Dict[str, list] = {
            'total_processed': 0,
            'keypoint_counts': [],
            'avg_scores': [],
            'spatial_coverage': [],
        }

    def _init_backend(self, name: str):
        if name == 'disk':
            b = _DISKBackend(self.max_keypoints, self.device)
            if b.available:
                return b
            print("[WARN] DISK unavailable, falling back to SIFT")
            return _SIFTBackend(self.max_keypoints)
        elif name == 'superpoint':
            b = _SuperPointBackend(self.max_keypoints, self.device)
            if b.available:
                return b
            print("[WARN] KeyNet+HardNet unavailable, falling back to DISK")
            b2 = _DISKBackend(self.max_keypoints, self.device)
            return b2 if b2.available else _SIFTBackend(self.max_keypoints)
        elif name == 'dedode':
            b = _DeDoDeMatcher(self.max_keypoints, self.device)
            if b.available:
                return b
            print("[WARN] DeDoDe unavailable, falling back to DISK")
            b2 = _DISKBackend(self.max_keypoints, self.device)
            return b2 if b2.available else _SIFTBackend(self.max_keypoints)
        else:  # 'sift'
            return _SIFTBackend(self.max_keypoints)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Extract keypoints and descriptors.

        Args:
            image: BGR or grayscale uint8 numpy array.
            mask:  Optional binary mask (foreground > 0).

        Returns:
            dict with 'keypoints' (N,2), 'descriptors' (N,256), 'scores' (N,).
        """
        self.stats['total_processed'] += 1

        # Normalise to RGB
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = image[:, :, ::-1].copy()   # BGR → RGB

        result = self._backend.extract(rgb, mask)

        # Record stats
        n_kp = len(result['keypoints'])
        self.stats['keypoint_counts'].append(n_kp)
        if n_kp > 0:
            self.stats['avg_scores'].append(float(np.mean(result['scores'])))
            h, w = image.shape[:2]
            if n_kp > 1:
                kp = result['keypoints']
                coverage = ((kp[:, 0].max() - kp[:, 0].min()) / w *
                            (kp[:, 1].max() - kp[:, 1].min()) / h)
            else:
                coverage = 0.0
            self.stats['spatial_coverage'].append(float(coverage))

        return result

    def visualize(self, image: np.ndarray,
                  keypoints: np.ndarray,
                  scores: Optional[np.ndarray] = None,
                  output_path: Optional[str] = None) -> np.ndarray:
        """Draw keypoints on image. Returns BGR visualization."""
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        if scores is not None and len(scores) > 0:
            s_min, s_max = scores.min(), scores.max()
            s_norm = (scores - s_min) / (s_max - s_min + 1e-8)
        else:
            s_norm = np.ones(len(keypoints))

        for i, (x, y) in enumerate(keypoints):
            color = (
                int(255 * (1 - s_norm[i])),
                0,
                int(255 * s_norm[i]),
            ) if i < len(s_norm) else (0, 255, 0)
            cv2.circle(vis, (int(x), int(y)), 3, color, -1)
            cv2.circle(vis, (int(x), int(y)), 4, (255, 255, 255), 1)

        cv2.putText(vis,
                    f"Keypoints: {len(keypoints)} ({self.method_name})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if output_path:
            cv2.imwrite(str(output_path), vis)
        return vis

    def get_stats(self) -> Dict:
        counts = self.stats['keypoint_counts']
        if not counts:
            return {
                'total_processed': self.stats['total_processed'],
                'method': self.method_name,
                'max_keypoints_setting': self.max_keypoints,
                'detection_threshold': self.detection_threshold,
                'nms_radius': self.nms_radius,
                'keypoint_counts': {'mean': 0, 'std': 0, 'min': 0,
                                    'max': 0, 'median': 0, 'total': 0},
                'detection_scores': {'mean': 0},
                'spatial_coverage': {'mean': 0, 'std': 0},
            }
        return {
            'total_processed': self.stats['total_processed'],
            'method': self.method_name,
            'max_keypoints_setting': self.max_keypoints,
            'detection_threshold': self.detection_threshold,
            'nms_radius': self.nms_radius,
            'keypoint_counts': {
                'mean':   float(np.mean(counts)),
                'std':    float(np.std(counts)),
                'min':    int(np.min(counts)),
                'max':    int(np.max(counts)),
                'median': float(np.median(counts)),
                'total':  int(np.sum(counts)),
            },
            'detection_scores': {
                'mean': (float(np.mean(self.stats['avg_scores']))
                         if self.stats['avg_scores'] else 0),
            },
            'spatial_coverage': {
                'mean': (float(np.mean(self.stats['spatial_coverage']))
                         if self.stats['spatial_coverage'] else 0),
                'std':  (float(np.std(self.stats['spatial_coverage']))
                         if self.stats['spatial_coverage'] else 0),
            },
        }


# ---------------------------------------------------------------------------
# Public: MultiExtractor  (runs all four backends in parallel)
# ---------------------------------------------------------------------------

class MultiExtractor:
    """
    Run all four Kornia/SIFT backends in parallel on the same image.

    Returns a dict keyed by backend name with timing and results.
    Backends that fail silently return empty results.
    """

    BACKEND_NAMES: Tuple[str, ...] = ('disk', 'superpoint', 'dedode', 'sift')

    def __init__(self,
                 max_keypoints: int = 256,
                 device: Optional[torch.device] = None,
                 backends: Optional[Tuple[str, ...]] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_keypoints = max_keypoints
        target = backends or self.BACKEND_NAMES

        print("[MultiExtractor] Initialising backends...")
        self.extractors: Dict[str, SuperPointExtractor] = {}
        for name in target:
            try:
                ext = SuperPointExtractor(
                    max_keypoints=max_keypoints,
                    device=self.device,
                    backend=name,
                )
                self.extractors[name] = ext
            except Exception as exc:
                warnings.warn(f"Could not init backend '{name}': {exc}")

        print(f"[MultiExtractor] Ready: {list(self.extractors.keys())}")

    def extract_parallel(self,
                         image: np.ndarray,
                         mask: Optional[np.ndarray] = None,
                         max_workers: int = 4
                         ) -> Dict[str, Dict]:
        """
        Extract keypoints from all backends in parallel threads.

        Args:
            image:       BGR or grayscale uint8 image.
            mask:        Optional binary mask.
            max_workers: ThreadPoolExecutor workers.

        Returns:
            dict: {backend_name: {'keypoints', 'descriptors', 'scores',
                                  'time_ms', 'method_name'}}
        """
        results: Dict[str, Dict] = {}

        def _run(name: str, ext: SuperPointExtractor):
            t0 = time.perf_counter()
            try:
                r = ext.extract(image, mask)
            except Exception as exc:
                warnings.warn(f"Backend '{name}' failed: {exc}")
                r = _empty_result(256)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return name, {**r,
                          'time_ms': elapsed_ms,
                          'method_name': ext.method_name}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run, name, ext): name
                       for name, ext in self.extractors.items()}
            for fut in concurrent.futures.as_completed(futures):
                name, res = fut.result()
                results[name] = res

        return results

    def extract_best(self,
                     image: np.ndarray,
                     mask: Optional[np.ndarray] = None,
                     criterion: str = 'count') -> Tuple[str, Dict]:
        """
        Extract from all backends, return the one with best keypoint count
        (or highest mean score if criterion='score').
        """
        all_res = self.extract_parallel(image, mask)
        if not all_res:
            return 'sift', _empty_result(256)

        if criterion == 'score':
            best = max(all_res,
                       key=lambda k: float(np.mean(all_res[k]['scores']))
                       if len(all_res[k]['scores']) else 0)
        else:  # 'count'
            best = max(all_res,
                       key=lambda k: len(all_res[k]['keypoints']))
        return best, all_res[best]
