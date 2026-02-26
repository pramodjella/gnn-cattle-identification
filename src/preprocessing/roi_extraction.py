"""
ROI Extraction Module
=====================
Handles muzzle region-of-interest extraction from cattle images.
Since the Zenodo dataset already contains cropped muzzle images,
this module primarily handles resizing and standardization.

For real-world deployment, this would use YOLOv8 for muzzle detection.
"""

import cv2
import numpy as np
from pathlib import Path


class ROIExtractor:
    """Extract and standardize muzzle ROI from images."""
    
    def __init__(self, target_size=256):
        """
        Args:
            target_size: Output image size (square, target_size x target_size)
        """
        self.target_size = target_size
        self.stats = {
            'total_processed': 0,
            'original_sizes': [],
            'aspect_ratios': [],
        }
    
    def extract(self, image):
        """
        Extract and standardize the muzzle ROI.
        
        For the Zenodo dataset (already cropped), this performs:
        1. Center crop to square aspect ratio
        2. Resize to target_size x target_size
        
        Args:
            image: Input image (BGR, numpy array)
            
        Returns:
            roi: Standardized ROI image
        """
        h, w = image.shape[:2]
        
        # Record stats
        self.stats['total_processed'] += 1
        self.stats['original_sizes'].append((w, h))
        self.stats['aspect_ratios'].append(w / h)
        
        # Center crop to square
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        cropped = image[start_y:start_y + min_dim, start_x:start_x + min_dim]
        
        # Resize to target size
        roi = cv2.resize(cropped, (self.target_size, self.target_size), 
                         interpolation=cv2.INTER_AREA)
        
        return roi
    
    def extract_with_yolo(self, image, model=None):
        """
        Extract muzzle ROI using YOLO detection (for real-world use).
        Falls back to center-crop if no detection.
        
        Args:
            image: Input image
            model: YOLO model (optional, for future use)
            
        Returns:
            roi: Extracted ROI
        """
        # Placeholder for YOLO-based detection
        # In production, this would:
        # 1. Run YOLOv8 inference on the image
        # 2. Find the muzzle bounding box
        # 3. Crop with padding
        # 4. Resize to target_size
        
        # For now, fall back to center crop
        return self.extract(image)
    
    def get_stats(self):
        """Return processing statistics."""
        if not self.stats['original_sizes']:
            return self.stats
        
        widths = [s[0] for s in self.stats['original_sizes']]
        heights = [s[1] for s in self.stats['original_sizes']]
        
        return {
            'total_processed': self.stats['total_processed'],
            'target_size': self.target_size,
            'original_width': {
                'mean': float(np.mean(widths)),
                'std': float(np.std(widths)),
                'min': int(np.min(widths)),
                'max': int(np.max(widths)),
            },
            'original_height': {
                'mean': float(np.mean(heights)),
                'std': float(np.std(heights)),
                'min': int(np.min(heights)),
                'max': int(np.max(heights)),
            },
            'aspect_ratios': {
                'mean': float(np.mean(self.stats['aspect_ratios'])),
                'std': float(np.std(self.stats['aspect_ratios'])),
            },
            'method': 'center_crop_resize (dataset pre-cropped)',
        }
