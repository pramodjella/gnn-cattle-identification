"""
Muzzle Segmentation Module
===========================
Generates binary masks to isolate muzzle texture regions from background.

Uses Otsu thresholding with morphological operations as a practical
segmentation approach. This provides an effective baseline without
requiring annotated segmentation masks for U-Net training.

For production, could be replaced with a trained U-Net model.
"""

import cv2
import numpy as np


class MuzzleSegmenter:
    """Generate binary segmentation masks for muzzle images."""
    
    def __init__(self, morph_kernel_size=5, min_area_ratio=0.1):
        """
        Args:
            morph_kernel_size: Kernel size for morphological operations
            min_area_ratio: Minimum connected component area as fraction of image
        """
        self.morph_kernel_size = morph_kernel_size
        self.min_area_ratio = min_area_ratio
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morph_kernel_size, morph_kernel_size)
        )
        self.stats = {
            'total_processed': 0,
            'mask_coverage': [],
            'num_components': [],
        }
    
    def segment(self, image):
        """
        Generate binary mask for the muzzle texture region.
        
        Pipeline:
        1. Convert to grayscale
        2. Gaussian blur for noise reduction
        3. Otsu thresholding
        4. Morphological closing (fill small holes)
        5. Morphological opening (remove small noise)
        6. Keep largest connected component
        
        Args:
            image: Input image (BGR, numpy array)
            
        Returns:
            mask: Binary mask (0 or 255, same size as input)
        """
        self.stats['total_processed'] += 1
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological closing (fill holes in foreground)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        
        # Morphological opening (remove small noise)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.kernel, iterations=2)
        
        # Keep largest connected component
        mask = self._keep_largest_component(opened)
        
        # Fill any remaining holes
        mask = self._fill_holes(mask)
        
        # Record stats
        h, w = mask.shape
        coverage = float(np.sum(mask > 0)) / (h * w)
        self.stats['mask_coverage'].append(coverage)
        
        return mask
    
    def _keep_largest_component(self, binary_mask):
        """Keep only the largest connected component."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        self.stats['num_components'].append(num_labels - 1)  # Exclude background
        
        if num_labels <= 1:
            return binary_mask
        
        # Find largest component (excluding background label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        
        # Create mask with only the largest component
        mask = np.zeros_like(binary_mask)
        mask[labels == largest_label] = 255
        
        return mask
    
    def _fill_holes(self, mask):
        """Fill holes in the binary mask using flood fill."""
        h, w = mask.shape
        
        # Create a copy with 2-pixel border for flood fill
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Flood fill from (0, 0) to find background
        inv_mask = cv2.bitwise_not(mask)
        cv2.floodFill(inv_mask, flood_mask, (0, 0), 0)
        
        # Combine original mask with filled version
        filled = mask | inv_mask
        
        return filled
    
    def apply_mask(self, image, mask):
        """
        Apply binary mask to an image.
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            masked: Image with background set to 0
        """
        if len(image.shape) == 3:
            mask_3ch = cv2.merge([mask, mask, mask])
            return cv2.bitwise_and(image, mask_3ch)
        else:
            return cv2.bitwise_and(image, mask)
    
    def get_stats(self):
        """Return segmentation statistics."""
        if not self.stats['mask_coverage']:
            return self.stats
        
        return {
            'total_processed': self.stats['total_processed'],
            'method': 'Otsu + Morphological Operations',
            'morph_kernel_size': self.morph_kernel_size,
            'mask_coverage': {
                'mean': float(np.mean(self.stats['mask_coverage'])),
                'std': float(np.std(self.stats['mask_coverage'])),
                'min': float(np.min(self.stats['mask_coverage'])),
                'max': float(np.max(self.stats['mask_coverage'])),
            },
            'connected_components': {
                'mean': float(np.mean(self.stats['num_components'])),
                'max': int(np.max(self.stats['num_components'])),
            },
        }
