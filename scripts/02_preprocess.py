"""
Script 02: Preprocessing Pipeline
===================================
Applies the full preprocessing pipeline to all dataset images:
1. ROI extraction (center crop + resize)
2. CLAHE contrast enhancement
3. Segmentation mask generation

Saves preprocessed images, masks, and preprocessing statistics.

Input:  data/raw/ (original images)
Output: data/preprocessed/ (enhanced images + masks)
Stats:  outputs/stats/preprocessing_stats.json
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, setup_logging, set_seed, Timer
from src.preprocessing.roi_extraction import ROIExtractor
from src.preprocessing.enhancement import CLAHEEnhancer
from src.preprocessing.segmentation import MuzzleSegmenter


def create_comparison_image(original, enhanced, mask, masked):
    """Create a side-by-side comparison image for visualization."""
    h, w = original.shape[:2]
    
    # Ensure all images are BGR
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if len(enhanced.shape) == 2:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if len(masked.shape) == 2:
        masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    
    # Stack horizontally
    comparison = np.hstack([original, enhanced, mask, masked])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ['Original', 'CLAHE', 'Mask', 'Masked']
    for i, label in enumerate(labels):
        x = i * w + 10
        cv2.putText(comparison, label, (x, 25), font, 0.6, (0, 255, 0), 2)
    
    return comparison


def preprocess_image(image_path, roi_extractor, enhancer, segmenter):
    """Apply full preprocessing pipeline to a single image."""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None, None, None
    
    # Step 1: ROI extraction
    roi = roi_extractor.extract(image)
    
    # Step 2: CLAHE enhancement
    enhanced = enhancer.enhance(roi)
    
    # Step 3: Segmentation
    mask = segmenter.segment(enhanced)
    
    # Step 4: Apply mask
    masked = segmenter.apply_mask(enhanced, mask)
    
    return roi, enhanced, mask, masked


def main():
    """Main preprocessing pipeline."""
    print("=" * 70)
    print("PHASE 2: Preprocessing Pipeline")
    print("=" * 70)
    
    # Load config
    config = load_config()
    set_seed(config['project']['seed'])
    logger = setup_logging(config['outputs']['log_dir'], "02_preprocess")
    
    # Setup paths
    raw_dir = PROJECT_ROOT / config['dataset']['raw_dir']
    processed_dir = PROJECT_ROOT / config['dataset']['processed_dir']
    stats_dir = PROJECT_ROOT / config['outputs']['stats_dir']
    figure_dir = PROJECT_ROOT / config['outputs']['figure_dir']
    
    ensure_dirs(
        str(processed_dir / "images"),
        str(processed_dir / "masks"),
        str(stats_dir),
        str(figure_dir / "preprocessing"),
    )
    
    # Load split information
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_file = processed_dir / f"{split_name}_split.json"
        if split_file.exists():
            with open(split_file, 'r') as f:
                splits[split_name] = json.load(f)
        else:
            print(f"[WARNING] Split file not found: {split_file}")
    
    if not splits:
        # If no splits found, process all images in raw directory
        print("[INFO] No split files found, processing all images in raw directory")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        all_images = []
        for ext in image_extensions:
            all_images.extend(raw_dir.rglob(f'*{ext}'))
            all_images.extend(raw_dir.rglob(f'*{ext.upper()}'))
        all_images = list(set(all_images))
        
        # Create a dummy split with all images
        splits = {'all': [{'image_path': str(p), 'animal_id': p.parent.name} for p in all_images]}
    
    # Initialize preprocessing modules
    prep_config = config['preprocessing']
    roi_extractor = ROIExtractor(target_size=prep_config['image_size'])
    enhancer = CLAHEEnhancer(
        clip_limit=prep_config['clahe']['clip_limit'],
        tile_grid_size=tuple(prep_config['clahe']['tile_grid_size'])
    )
    segmenter = MuzzleSegmenter(
        morph_kernel_size=prep_config['segmentation']['morph_kernel_size'],
        min_area_ratio=prep_config['segmentation']['min_area_ratio']
    )
    
    # Process all images
    total_images = sum(len(v) for v in splits.values())
    print(f"\n[INFO] Processing {total_images} images...")
    
    processed_count = 0
    failed_count = 0
    sample_comparisons = []
    
    with Timer("Preprocessing") as timer:
        for split_name, split_data in splits.items():
            split_img_dir = processed_dir / "images" / split_name
            split_mask_dir = processed_dir / "masks" / split_name
            ensure_dirs(str(split_img_dir), str(split_mask_dir))
            
            for item in tqdm(split_data, desc=f"Processing {split_name}"):
                image_path = Path(item['image_path'])
                animal_id = item['animal_id']
                
                # Create output directory per animal
                animal_img_dir = split_img_dir / animal_id
                animal_mask_dir = split_mask_dir / animal_id
                ensure_dirs(str(animal_img_dir), str(animal_mask_dir))
                
                # Output filename
                out_name = f"{image_path.stem}.png"
                img_out_path = animal_img_dir / out_name
                mask_out_path = animal_mask_dir / out_name
                
                # Skip if already processed
                if img_out_path.exists() and mask_out_path.exists():
                    processed_count += 1
                    continue
                
                # Process
                roi, enhanced, mask, masked = preprocess_image(
                    image_path, roi_extractor, enhancer, segmenter
                )
                
                if roi is None:
                    failed_count += 1
                    logger.warning(f"Failed to process: {image_path}")
                    continue
                
                # Save preprocessed image and mask
                cv2.imwrite(str(img_out_path), enhanced)
                cv2.imwrite(str(mask_out_path), mask)
                
                processed_count += 1
                
                # Save sample comparisons (first 10 per split)
                if len(sample_comparisons) < 10:
                    comparison = create_comparison_image(roi, enhanced, mask, masked)
                    sample_comparisons.append((comparison, animal_id, image_path.stem))
    
    # Save sample comparison images
    print(f"\n[INFO] Saving sample comparison images...")
    for i, (comparison, animal_id, stem) in enumerate(sample_comparisons):
        comp_path = figure_dir / "preprocessing" / f"comparison_{i:03d}_{animal_id}.png"
        cv2.imwrite(str(comp_path), comparison)
    
    # Collect and save statistics
    preprocessing_stats = {
        'pipeline_info': {
            'processing_time_seconds': timer.elapsed,
            'total_processed': processed_count,
            'total_failed': failed_count,
            'image_size': prep_config['image_size'],
        },
        'roi_extraction': roi_extractor.get_stats(),
        'clahe_enhancement': enhancer.get_stats(),
        'segmentation': segmenter.get_stats(),
    }
    
    stats_path = str(stats_dir / "preprocessing_stats.json")
    save_stats(preprocessing_stats, stats_path)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("PREPROCESSING STATISTICS")
    print(f"{'=' * 70}")
    print(f"  Total Processed:     {processed_count}")
    print(f"  Total Failed:        {failed_count}")
    print(f"  Processing Time:     {timer.elapsed:.1f}s")
    print(f"  Image Size:          {prep_config['image_size']}×{prep_config['image_size']}")
    
    enh_stats = enhancer.get_stats()
    if 'contrast_rms' in enh_stats:
        print(f"\n  CLAHE Enhancement:")
        print(f"    Contrast (RMS):    {enh_stats['contrast_rms']['before_mean']:.1f} -> {enh_stats['contrast_rms']['after_mean']:.1f} ({enh_stats['contrast_rms']['improvement_pct']})")
        print(f"    Entropy:           {enh_stats['entropy']['before_mean']:.2f} -> {enh_stats['entropy']['after_mean']:.2f} ({enh_stats['entropy']['improvement_pct']})")
    
    seg_stats = segmenter.get_stats()
    if 'mask_coverage' in seg_stats and isinstance(seg_stats['mask_coverage'], dict):
        print(f"\n  Segmentation:")
        print(f"    Mask Coverage:     {seg_stats['mask_coverage']['mean']:.1%} ± {seg_stats['mask_coverage']['std']:.1%}")
    
    print(f"{'=' * 70}")
    print(f"\n[SUCCESS] [OK] Phase 2 complete!")
    print(f"  Preprocessed images: {processed_dir / 'images'}")
    print(f"  Segmentation masks:  {processed_dir / 'masks'}")
    print(f"  Statistics:          {stats_path}")
    print(f"  Sample comparisons:  {figure_dir / 'preprocessing'}")
    
    return preprocessing_stats


if __name__ == "__main__":
    main()
