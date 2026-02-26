"""
CLAHE Contrast Enhancement Module
==================================
Applies Contrast Limited Adaptive Histogram Equalization to enhance
bead/ridge texture patterns on cattle muzzle images.

CLAHE enhances local contrast while limiting noise amplification,
making it ideal for highlighting fine surface textures.
"""

import cv2
import numpy as np
from skimage.measure import shannon_entropy


class CLAHEEnhancer:
    """Apply CLAHE enhancement to muzzle images."""
    
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        """
        Args:
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        self.stats = {
            'total_processed': 0,
            'contrast_before': [],
            'contrast_after': [],
            'entropy_before': [],
            'entropy_after': [],
        }
    
    def _compute_rms_contrast(self, image_gray):
        """Compute RMS contrast of a grayscale image."""
        return float(np.std(image_gray.astype(np.float64)))
    
    def _compute_entropy(self, image_gray):
        """Compute Shannon entropy of a grayscale image."""
        return float(shannon_entropy(image_gray))
    
    def enhance(self, image):
        """
        Apply CLAHE enhancement to an image.
        
        Args:
            image: Input image (BGR or grayscale, numpy array)
            
        Returns:
            enhanced: CLAHE-enhanced image (same color space as input)
        """
        self.stats['total_processed'] += 1
        
        if len(image.shape) == 2:
            # Grayscale image
            gray = image
        else:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            gray = lab[:, :, 0]
        
        # Compute before-enhancement metrics
        self.stats['contrast_before'].append(self._compute_rms_contrast(gray))
        self.stats['entropy_before'].append(self._compute_entropy(gray))
        
        # Apply CLAHE
        enhanced_channel = self.clahe.apply(gray)
        
        # Compute after-enhancement metrics
        self.stats['contrast_after'].append(self._compute_rms_contrast(enhanced_channel))
        self.stats['entropy_after'].append(self._compute_entropy(enhanced_channel))
        
        if len(image.shape) == 2:
            return enhanced_channel
        else:
            # Replace L channel with enhanced version
            lab[:, :, 0] = enhanced_channel
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return enhanced
    
    def get_stats(self):
        """Return enhancement statistics for paper reporting."""
        if not self.stats['contrast_before']:
            return self.stats
        
        contrast_improvement = [
            (a - b) / b * 100 
            for a, b in zip(self.stats['contrast_after'], self.stats['contrast_before'])
            if b > 0
        ]
        
        entropy_improvement = [
            (a - b) / b * 100 
            for a, b in zip(self.stats['entropy_after'], self.stats['entropy_before'])
            if b > 0
        ]
        
        return {
            'total_processed': self.stats['total_processed'],
            'clahe_params': {
                'clip_limit': self.clip_limit,
                'tile_grid_size': list(self.tile_grid_size),
            },
            'contrast_rms': {
                'before_mean': float(np.mean(self.stats['contrast_before'])),
                'after_mean': float(np.mean(self.stats['contrast_after'])),
                'improvement_pct': f"{np.mean(contrast_improvement):.1f}%",
            },
            'entropy': {
                'before_mean': float(np.mean(self.stats['entropy_before'])),
                'after_mean': float(np.mean(self.stats['entropy_after'])),
                'improvement_pct': f"{np.mean(entropy_improvement):.1f}%",
            },
        }
