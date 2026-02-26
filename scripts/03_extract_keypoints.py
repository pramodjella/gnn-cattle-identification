"""
Script 03: Keypoint Extraction
================================
Extracts SuperPoint keypoints and 256-dim descriptors from all
preprocessed muzzle images. Saves keypoint data and visualizations.

Input:  data/preprocessed/images/
Output: data/preprocessed/keypoints/ (per-image .npz files)
Stats:  outputs/stats/keypoint_stats.json
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, setup_logging, set_seed, Timer
from src.features.superpoint import SuperPointExtractor


def main():
    print("=" * 70)
    print("PHASE 3: Keypoint Detection & Description")
    print("=" * 70)
    
    config = load_config()
    set_seed(config['project']['seed'])
    logger = setup_logging(config['outputs']['log_dir'], "03_extract_keypoints")
    
    processed_dir = PROJECT_ROOT / config['dataset']['processed_dir']
    stats_dir = PROJECT_ROOT / config['outputs']['stats_dir']
    figure_dir = PROJECT_ROOT / config['outputs']['figure_dir']
    
    kp_dir = processed_dir / "keypoints"
    ensure_dirs(str(kp_dir), str(stats_dir), str(figure_dir / "keypoints"))
    
    # Initialize SuperPoint
    kp_config = config['keypoints']
    extractor = SuperPointExtractor(
        max_keypoints=kp_config['max_keypoints'],
        detection_threshold=kp_config['detection_threshold'],
        nms_radius=kp_config['nms_radius'],
    )
    
    # Find all preprocessed images
    image_dir = processed_dir / "images"
    mask_dir = processed_dir / "masks"
    
    all_images = []
    for split_dir in image_dir.iterdir():
        if split_dir.is_dir():
            for animal_dir in split_dir.iterdir():
                if animal_dir.is_dir():
                    for img_path in animal_dir.glob("*.png"):
                        # Corresponding mask
                        mask_path = mask_dir / split_dir.name / animal_dir.name / img_path.name
                        all_images.append({
                            'image_path': img_path,
                            'mask_path': mask_path if mask_path.exists() else None,
                            'split': split_dir.name,
                            'animal_id': animal_dir.name,
                        })
    
    print(f"[INFO] Found {len(all_images)} preprocessed images")
    
    if len(all_images) == 0:
        print("[WARNING] No preprocessed images found. Run 02_preprocess.py first.")
        return
    
    # Extract keypoints
    sample_vis_count = 0
    max_vis = 20  # Save 20 sample visualizations
    
    with Timer("Keypoint Extraction") as timer:
        for item in tqdm(all_images, desc="Extracting keypoints"):
            img_path = item['image_path']
            
            # Output path
            kp_out_dir = kp_dir / item['split'] / item['animal_id']
            ensure_dirs(str(kp_out_dir))
            kp_out_path = kp_out_dir / f"{img_path.stem}.npz"
            
            # Skip if already extracted
            if kp_out_path.exists():
                continue
            
            # Load image and mask
            image = cv2.imread(str(img_path))
            mask = None
            if item['mask_path'] and item['mask_path'].exists():
                mask = cv2.imread(str(item['mask_path']), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                logger.warning(f"Failed to load: {img_path}")
                continue
            
            # Extract keypoints
            result = extractor.extract(image, mask=mask)
            
            # Save keypoint data
            np.savez_compressed(
                str(kp_out_path),
                keypoints=result['keypoints'],
                descriptors=result['descriptors'],
                scores=result['scores'],
                animal_id=item['animal_id'],
                image_path=str(img_path),
            )
            
            # Save sample visualizations
            if sample_vis_count < max_vis and len(result['keypoints']) > 0:
                vis = extractor.visualize(
                    image, result['keypoints'], result['scores'],
                    output_path=str(figure_dir / "keypoints" / f"kp_{sample_vis_count:03d}_{item['animal_id']}.png")
                )
                sample_vis_count += 1
    
    # Save stats
    kp_stats = extractor.get_stats()
    kp_stats['processing_time_seconds'] = timer.elapsed
    
    stats_path = str(stats_dir / "keypoint_stats.json")
    save_stats(kp_stats, stats_path)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("KEYPOINT DETECTION STATISTICS")
    print(f"{'=' * 70}")
    print(f"  Method:              {kp_stats['method']}")
    print(f"  Total Images:        {kp_stats['total_processed']}")
    print(f"  Processing Time:     {timer.elapsed:.1f}s")
    print(f"  Keypoints/Image:     {kp_stats['keypoint_counts']['mean']:.1f} ± {kp_stats['keypoint_counts']['std']:.1f}")
    print(f"  Min Keypoints:       {kp_stats['keypoint_counts']['min']}")
    print(f"  Max Keypoints:       {kp_stats['keypoint_counts']['max']}")
    print(f"  Total Keypoints:     {kp_stats['keypoint_counts']['total']}")
    print(f"  Spatial Coverage:    {kp_stats['spatial_coverage']['mean']:.1%}")
    print(f"{'=' * 70}")
    print(f"\n[SUCCESS] [OK] Phase 3 complete! Keypoints saved to {kp_dir}")
    
    return kp_stats


if __name__ == "__main__":
    main()
