"""
SuperPoint Keypoint Detector and Descriptor Extractor
=====================================================
Uses the SuperPoint neural network for self-supervised keypoint detection
and 256-dimensional descriptor computation on cattle muzzle images.

SuperPoint detects biological landmarks: bead centroids (protuberances)
and ridge endpoints (grooves) on the muzzle surface.
"""

import cv2
import torch
import numpy as np
from pathlib import Path


class SuperPointExtractor:
    """Extract keypoints and descriptors using SuperPoint."""
    
    def __init__(self, max_keypoints=512, detection_threshold=0.005, 
                 nms_radius=4, device=None):
        """
        Args:
            max_keypoints: Maximum number of keypoints to detect
            detection_threshold: Detection confidence threshold
            nms_radius: Non-maximum suppression radius
            device: Computation device (auto-detected if None)
        """
        self.max_keypoints = max_keypoints
        self.detection_threshold = detection_threshold
        self.nms_radius = nms_radius
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SuperPoint model
        self.model = self._load_model()
        
        self.stats = {
            'total_processed': 0,
            'keypoint_counts': [],
            'avg_scores': [],
            'spatial_coverage': [],
        }
    
    def _load_model(self):
        """Load SuperPoint model using kornia."""
        try:
            from kornia.feature import SuperPoint as KorniaSuperPoint
            
            model = KorniaSuperPoint(
                num_features=self.max_keypoints,
                detection_threshold=self.detection_threshold,
                nms_radius=self.nms_radius,
            ).to(self.device)
            model.eval()
            print(f"[INFO] SuperPoint loaded via kornia on {self.device}")
            return model
        except ImportError:
            print("[WARNING] kornia not available, falling back to OpenCV SIFT")
            return None
    
    def extract(self, image, mask=None):
        """
        Extract keypoints and descriptors from an image.
        
        Args:
            image: Input image (BGR or grayscale, numpy array)
            mask: Optional binary mask to restrict keypoint detection
            
        Returns:
            dict with:
                'keypoints': (N, 2) array of (x, y) positions
                'descriptors': (N, 256) array of descriptors
                'scores': (N,) array of detection scores
        """
        self.stats['total_processed'] += 1
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if self.model is not None:
            result = self._extract_superpoint(gray, mask)
        else:
            result = self._extract_sift(gray, mask)
        
        # Record stats
        n_kp = len(result['keypoints'])
        self.stats['keypoint_counts'].append(n_kp)
        
        if n_kp > 0:
            self.stats['avg_scores'].append(float(np.mean(result['scores'])))
            
            # Compute spatial coverage (fraction of image covered by keypoints)
            h, w = gray.shape
            if n_kp > 1:
                kp = result['keypoints']
                x_range = (np.max(kp[:, 0]) - np.min(kp[:, 0])) / w
                y_range = (np.max(kp[:, 1]) - np.min(kp[:, 1])) / h
                coverage = x_range * y_range
            else:
                coverage = 0.0
            self.stats['spatial_coverage'].append(float(coverage))
        
        return result
    
    def _extract_superpoint(self, gray, mask=None):
        """Extract using SuperPoint via kornia."""
        # Prepare input tensor
        h, w = gray.shape
        img_tensor = torch.from_numpy(gray).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)
        
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Extract results
        keypoints = output['keypoints'][0].cpu().numpy()  # (N, 2)
        scores = output['keypoint_scores'][0].cpu().numpy()  # (N,)
        descriptors = output['descriptors'][0].cpu().numpy()  # (N, 256)
        
        # Apply mask if provided
        if mask is not None and len(keypoints) > 0:
            valid = []
            for i, (x, y) in enumerate(keypoints):
                ix, iy = int(round(x)), int(round(y))
                if 0 <= iy < h and 0 <= ix < w and mask[iy, ix] > 0:
                    valid.append(i)
            
            if valid:
                keypoints = keypoints[valid]
                scores = scores[valid]
                descriptors = descriptors[valid]
            else:
                keypoints = np.zeros((0, 2), dtype=np.float32)
                scores = np.zeros(0, dtype=np.float32)
                descriptors = np.zeros((0, 256), dtype=np.float32)
        
        # Limit to max keypoints (keep highest scoring)
        if len(keypoints) > self.max_keypoints:
            top_idx = np.argsort(scores)[::-1][:self.max_keypoints]
            keypoints = keypoints[top_idx]
            scores = scores[top_idx]
            descriptors = descriptors[top_idx]
        
        return {
            'keypoints': keypoints.astype(np.float32),
            'descriptors': descriptors.astype(np.float32),
            'scores': scores.astype(np.float32),
        }
    
    def _extract_sift(self, gray, mask=None):
        """Fallback: Extract using SIFT."""
        sift = cv2.SIFT_create(nfeatures=self.max_keypoints)
        
        mask_uint8 = mask if mask is not None else None
        kps, descs = sift.detectAndCompute(gray, mask_uint8)
        
        if kps is None or len(kps) == 0:
            return {
                'keypoints': np.zeros((0, 2), dtype=np.float32),
                'descriptors': np.zeros((0, 128), dtype=np.float32),
                'scores': np.zeros(0, dtype=np.float32),
            }
        
        keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
        scores = np.array([kp.response for kp in kps], dtype=np.float32)
        
        # Pad SIFT descriptors from 128 to 256 dim for consistency
        if descs.shape[1] < 256:
            padding = np.zeros((descs.shape[0], 256 - descs.shape[1]), dtype=np.float32)
            descs = np.hstack([descs, padding])
        
        return {
            'keypoints': keypoints,
            'descriptors': descs.astype(np.float32),
            'scores': scores,
        }
    
    def visualize(self, image, keypoints, scores=None, output_path=None):
        """
        Visualize detected keypoints on the image.
        
        Args:
            image: Input image (BGR)
            keypoints: (N, 2) array of (x, y) positions
            scores: (N,) array of detection scores (for color coding)
            output_path: Path to save visualization
            
        Returns:
            vis: Visualization image
        """
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        if scores is not None:
            # Normalize scores for color mapping
            if len(scores) > 0:
                s_min, s_max = scores.min(), scores.max()
                if s_max > s_min:
                    s_norm = (scores - s_min) / (s_max - s_min)
                else:
                    s_norm = np.ones_like(scores)
            else:
                s_norm = np.array([])
        else:
            s_norm = np.ones(len(keypoints))
        
        for i, (x, y) in enumerate(keypoints):
            # Color: blue (low score) → red (high score)
            if i < len(s_norm):
                color = (
                    int(255 * (1 - s_norm[i])),  # B
                    0,                             # G
                    int(255 * s_norm[i]),           # R
                )
            else:
                color = (0, 255, 0)
            
            radius = 3
            cv2.circle(vis, (int(x), int(y)), radius, color, -1)
            cv2.circle(vis, (int(x), int(y)), radius + 1, (255, 255, 255), 1)
        
        # Add count text
        cv2.putText(vis, f"Keypoints: {len(keypoints)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(str(output_path), vis)
        
        return vis
    
    def get_stats(self):
        """Return keypoint detection statistics."""
        if not self.stats['keypoint_counts']:
            return self.stats
        
        counts = self.stats['keypoint_counts']
        
        return {
            'total_processed': self.stats['total_processed'],
            'method': 'SuperPoint' if self.model is not None else 'SIFT (fallback)',
            'max_keypoints_setting': self.max_keypoints,
            'detection_threshold': self.detection_threshold,
            'nms_radius': self.nms_radius,
            'keypoint_counts': {
                'mean': float(np.mean(counts)),
                'std': float(np.std(counts)),
                'min': int(np.min(counts)),
                'max': int(np.max(counts)),
                'median': float(np.median(counts)),
                'total': int(np.sum(counts)),
            },
            'detection_scores': {
                'mean': float(np.mean(self.stats['avg_scores'])) if self.stats['avg_scores'] else 0,
            },
            'spatial_coverage': {
                'mean': float(np.mean(self.stats['spatial_coverage'])) if self.stats['spatial_coverage'] else 0,
                'std': float(np.std(self.stats['spatial_coverage'])) if self.stats['spatial_coverage'] else 0,
            },
        }
