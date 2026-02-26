"""
Script 01: Download and Prepare the Beef Cattle Muzzle Dataset
=============================================================
Downloads the Zenodo Beef Cattle Muzzle Database (DOI: 10.5281/zenodo.6324361),
extracts it, organizes the directory structure, computes dataset statistics,
and creates train/val/test splits.

Dataset: 4923 muzzle images from 268 beef cattle
Source: https://zenodo.org/records/6324361

Output Stats Saved: outputs/stats/dataset_stats.json
"""

import os
import sys
import json
import shutil
import zipfile
import requests
import hashlib
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, setup_logging, set_seed


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return dest_path


def download_zenodo_dataset(record_id, raw_dir):
    """Download dataset from Zenodo using the API."""
    raw_dir = Path(raw_dir)
    ensure_dirs(str(raw_dir))
    
    zip_path = raw_dir / "BeefCattle_Muzzle_database.zip"
    
    if zip_path.exists():
        print(f"[INFO] Dataset zip already exists at {zip_path}")
        return zip_path
    
    # Zenodo API endpoint
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"[INFO] Fetching record metadata from Zenodo (Record ID: {record_id})...")
    
    response = requests.get(api_url)
    response.raise_for_status()
    record = response.json()
    
    print(f"[INFO] Record Title: {record.get('metadata', {}).get('title', 'N/A')}")
    print(f"[INFO] DOI: {record.get('doi', 'N/A')}")
    
    # Find the zip file in the record's files
    files = record.get('files', [])
    target_file = None
    for f in files:
        if f['key'].endswith('.zip'):
            target_file = f
            break
    
    if target_file is None:
        print("[ERROR] No zip file found in the Zenodo record.")
        print("[INFO] Available files:")
        for f in files:
            print(f"  - {f['key']} ({f['size'] / 1e6:.1f} MB)")
        raise FileNotFoundError("No zip file found in Zenodo record")
    
    download_url = target_file['links']['self']
    file_size_mb = target_file['size'] / 1e6
    print(f"[INFO] Downloading: {target_file['key']} ({file_size_mb:.1f} MB)")
    
    download_file(download_url, str(zip_path))
    
    # Verify checksum if available
    if 'checksum' in target_file:
        expected_checksum = target_file['checksum']
        algo, expected_hash = expected_checksum.split(':')
        print(f"[INFO] Verifying {algo} checksum...")
        
        hash_func = hashlib.new(algo)
        with open(zip_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        actual_hash = hash_func.hexdigest()
        
        if actual_hash == expected_hash:
            print("[INFO] [OK] Checksum verified successfully")
        else:
            print(f"[WARNING] Checksum mismatch! Expected {expected_hash}, got {actual_hash}")
    
    print(f"[INFO] [OK] Download complete: {zip_path}")
    return zip_path


def extract_dataset(zip_path, raw_dir):
    """Extract the dataset zip file."""
    raw_dir = Path(raw_dir)
    extracted_marker = raw_dir / ".extracted"
    
    if extracted_marker.exists():
        print("[INFO] Dataset already extracted")
        return
    
    print(f"[INFO] Extracting dataset from {zip_path}...")
    with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, str(raw_dir))
    
    # Create marker file
    extracted_marker.touch()
    print(f"[INFO] [OK] Extraction complete to {raw_dir}")


def organize_dataset(raw_dir):
    """
    Organize the dataset into a standardized structure.
    Returns dict mapping animal_id -> list of image paths.
    """
    raw_dir = Path(raw_dir)
    
    # Find all image files recursively
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(raw_dir.rglob(f'*{ext}'))
        all_images.extend(raw_dir.rglob(f'*{ext.upper()}'))
    
    # Remove duplicates
    all_images = list(set(all_images))
    
    print(f"[INFO] Found {len(all_images)} total image files")
    
    # Organize by parent folder (animal ID)
    animal_images = defaultdict(list)
    for img_path in all_images:
        # The parent folder name is the animal ID
        animal_id = img_path.parent.name
        # Skip if the parent is the raw_dir itself (not organized in folders)
        if str(img_path.parent) == str(raw_dir):
            # Try to extract animal ID from filename
            animal_id = img_path.stem.split('_')[0] if '_' in img_path.stem else img_path.stem.split('-')[0]
        animal_images[animal_id].append(str(img_path))
    
    # Sort images within each animal
    for animal_id in animal_images:
        animal_images[animal_id].sort()
    
    return dict(animal_images)


def create_splits(animal_images, config, seed=42):
    """
    Create train/val/test splits at the IMAGE level, stratified by animal.
    Each split contains images from all animals (not separating animals into splits).
    """
    import numpy as np
    
    set_seed(seed)
    
    split_ratios = config['dataset']['split_ratios']
    min_images = config['dataset'].get('min_images_per_animal', 3)
    
    # Filter animals with too few images
    filtered = {
        aid: imgs for aid, imgs in animal_images.items()
        if len(imgs) >= min_images
    }
    
    removed = len(animal_images) - len(filtered)
    if removed > 0:
        print(f"[INFO] Removed {removed} animals with fewer than {min_images} images")
    
    splits = {'train': [], 'val': [], 'test': []}
    animal_splits = {'train': [], 'val': [], 'test': []}
    
    for animal_id, images in sorted(filtered.items()):
        n = len(images)
        # Shuffle images for this animal
        shuffled = images.copy()
        np.random.shuffle(shuffled)
        
        n_train = max(1, int(n * split_ratios['train']))
        n_val = max(1, int(n * split_ratios['val']))
        n_test = n - n_train - n_val
        
        if n_test < 1:
            n_test = 1
            n_train = n - n_val - n_test
        
        train_imgs = shuffled[:n_train]
        val_imgs = shuffled[n_train:n_train + n_val]
        test_imgs = shuffled[n_train + n_val:]
        
        for img in train_imgs:
            splits['train'].append({'image_path': img, 'animal_id': animal_id})
        for img in val_imgs:
            splits['val'].append({'image_path': img, 'animal_id': animal_id})
        for img in test_imgs:
            splits['test'].append({'image_path': img, 'animal_id': animal_id})
        
        animal_splits['train'].append(animal_id)
        animal_splits['val'].append(animal_id)
        animal_splits['test'].append(animal_id)
    
    return splits, filtered


def compute_dataset_statistics(animal_images, splits):
    """Compute comprehensive dataset statistics for the paper."""
    
    # Basic counts
    total_images = sum(len(imgs) for imgs in animal_images.values())
    total_animals = len(animal_images)
    images_per_animal = [len(imgs) for imgs in animal_images.values()]
    
    stats = {
        "dataset_info": {
            "name": "Beef Cattle Muzzle Database",
            "source": "Zenodo (DOI: 10.5281/zenodo.6324361)",
            "total_images": total_images,
            "total_animals": total_animals,
            "images_per_animal": {
                "mean": float(np.mean(images_per_animal)),
                "std": float(np.std(images_per_animal)),
                "min": int(np.min(images_per_animal)),
                "max": int(np.max(images_per_animal)),
                "median": float(np.median(images_per_animal)),
            },
            "distribution": dict(Counter(images_per_animal).most_common()),
        },
        "splits": {
            split_name: {
                "num_images": len(split_data),
                "num_animals": len(set(item['animal_id'] for item in split_data)),
                "images_per_animal": {
                    "mean": float(np.mean([
                        sum(1 for item in split_data if item['animal_id'] == aid)
                        for aid in set(item['animal_id'] for item in split_data)
                    ])) if split_data else 0,
                }
            }
            for split_name, split_data in splits.items()
        },
        "paper_ready": {
            "total_samples": total_images,
            "num_classes": total_animals,
            "avg_samples_per_class": f"{np.mean(images_per_animal):.1f} ± {np.std(images_per_animal):.1f}",
            "train_size": len(splits['train']),
            "val_size": len(splits['val']),
            "test_size": len(splits['test']),
            "split_ratio": "70/15/15",
        }
    }
    
    return stats


def save_splits(splits, output_dir):
    """Save train/val/test splits to JSON files."""
    ensure_dirs(output_dir)
    
    for split_name, split_data in splits.items():
        filepath = os.path.join(output_dir, f"{split_name}_split.json")
        with open(filepath, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"[INFO] Saved {split_name} split ({len(split_data)} images) to {filepath}")


def main():
    """Main entry point for data download and preparation."""
    import numpy as np
    
    print("=" * 70)
    print("PHASE 1: Dataset Download and Preparation")
    print("=" * 70)
    
    # Load config
    config = load_config()
    set_seed(config['project']['seed'])
    
    # Setup logging
    logger = setup_logging(config['outputs']['log_dir'], "01_download_data")
    
    # Setup directories
    raw_dir = PROJECT_ROOT / config['dataset']['raw_dir']
    stats_dir = PROJECT_ROOT / config['outputs']['stats_dir']
    ensure_dirs(str(raw_dir), str(stats_dir))
    
    # Step 1: Download dataset
    print("\n--- Step 1: Downloading Dataset ---")
    record_id = config['dataset']['zenodo_record_id']
    
    try:
        zip_path = download_zenodo_dataset(record_id, str(raw_dir))
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        print(f"\n[INFO] MANUAL DOWNLOAD INSTRUCTIONS:")
        print(f"  1. Visit: https://zenodo.org/records/{record_id}")
        print(f"  2. Download 'BeefCattle_Muzzle_database.zip'")
        print(f"  3. Place it in: {raw_dir}")
        print(f"  4. Re-run this script")
        
        # Check if zip was manually placed
        zip_path = raw_dir / "BeefCattle_Muzzle_database.zip"
        if not zip_path.exists():
            # Look for any zip file in the directory
            zips = list(raw_dir.glob("*.zip"))
            if zips:
                zip_path = zips[0]
                print(f"\n[INFO] Found existing zip: {zip_path}")
            else:
                print("[ERROR] No zip file found. Exiting.")
                sys.exit(1)
    
    # Step 2: Extract dataset
    print("\n--- Step 2: Extracting Dataset ---")
    extract_dataset(zip_path, str(raw_dir))
    
    # Step 3: Organize dataset
    print("\n--- Step 3: Organizing Dataset ---")
    animal_images = organize_dataset(raw_dir)
    
    print(f"\n[INFO] Dataset Organization:")
    print(f"  Total animals: {len(animal_images)}")
    print(f"  Total images: {sum(len(v) for v in animal_images.values())}")
    
    # Step 4: Create train/val/test splits
    print("\n--- Step 4: Creating Train/Val/Test Splits ---")
    splits, filtered_animals = create_splits(animal_images, config)
    
    for split_name, split_data in splits.items():
        n_animals = len(set(item['animal_id'] for item in split_data))
        print(f"  {split_name}: {len(split_data)} images, {n_animals} animals")
    
    # Save splits
    split_dir = str(PROJECT_ROOT / config['dataset']['processed_dir'])
    save_splits(splits, split_dir)
    
    # Step 5: Compute and save statistics
    print("\n--- Step 5: Computing Dataset Statistics ---")
    stats = compute_dataset_statistics(filtered_animals, splits)
    
    stats_path = str(stats_dir / "dataset_stats.json")
    save_stats(stats, stats_path)
    print(f"[INFO] [OK] Dataset statistics saved to {stats_path}")
    
    # Print paper-ready statistics
    print("\n" + "=" * 70)
    print("PAPER-READY DATASET STATISTICS")
    print("=" * 70)
    ps = stats['paper_ready']
    print(f"  Dataset:               {stats['dataset_info']['name']}")
    print(f"  Source:                 {stats['dataset_info']['source']}")
    print(f"  Total Samples:         {ps['total_samples']}")
    print(f"  Number of Classes:     {ps['num_classes']}")
    print(f"  Avg Samples/Class:     {ps['avg_samples_per_class']}")
    print(f"  Train/Val/Test Split:  {ps['split_ratio']}")
    print(f"  Train Size:            {ps['train_size']}")
    print(f"  Validation Size:       {ps['val_size']}")
    print(f"  Test Size:             {ps['test_size']}")
    print("=" * 70)
    
    print("\n[SUCCESS] [OK] Phase 1 complete! Dataset is ready for preprocessing.")
    return stats


if __name__ == "__main__":
    main()
