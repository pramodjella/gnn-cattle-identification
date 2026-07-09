"""
Prepare an External Cross-Dataset (Zenodo -> folder-per-animal)
==============================================================
Downloads a public Zenodo dataset, extracts it, normalises it to the
folder-per-animal layout expected by `ExternalMuzzleImageDataset`, and
(optionally) crops muzzle regions from YOLO bounding-box labels.

Default target is the Pakistan cattle dataset (Zenodo record 10535934,
"Cows Frontal Face Dataset", 459 individuals, CC BY 4.0) — a distribution
distinct from the primary US-feedlot training set, ideal for zero-shot
cross-dataset transfer.

Typical use:
    # 1. download + extract (~13.9 GB; run on a machine with disk + bandwidth)
    python scripts/prepare_cross_dataset.py --zenodo-record 10535934 --download --extract

    # 2. normalise to folder-per-animal (auto-detects the per-animal level)
    python scripts/prepare_cross_dataset.py --organize \
        --src data/external/10535934/extracted \
        --out data/external/pakistan_muzzle

    # 3. (optional) if YOLO .txt labels sit next to images, crop the muzzle box
    python scripts/prepare_cross_dataset.py --organize --crop-muzzle \
        --src data/external/10535934/extracted \
        --out data/external/pakistan_muzzle_cropped

Then evaluate transfer:
    python scripts/evaluate_cross_dataset.py --data-root data/external/pakistan_muzzle
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def _download_resumable(url: str, out: Path, expected_size=None,
                        max_stalls: int = 15, chunk: int = 1 << 20):
    """Stream a URL to disk with HTTP Range resume, retries, and backoff.

    Survives dropped connections and transient DNS/network outages on large
    (10+ GB) files by re-requesting only the missing byte range and appending.
    Aborts only after ``max_stalls`` *consecutive* failures that make no
    progress (so a slow-but-advancing download continues indefinitely).
    """
    import time
    out.parent.mkdir(parents=True, exist_ok=True)
    stalls = 0
    last_size = -1
    while True:
        have = out.stat().st_size if out.exists() else 0
        if expected_size and have >= expected_size:
            return
        req = urllib.request.Request(url)
        if have:
            req.add_header('Range', f'bytes={have}-')
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                # 206 => partial (append); 200 => server ignored Range (restart).
                mode = 'ab'
                if have and resp.status == 200:
                    mode, have = 'wb', 0
                with open(out, mode) as fh:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        fh.write(buf)
                        have += len(buf)
            if expected_size and out.stat().st_size < expected_size:
                raise IOError("short read; will resume")
            print(f"      done ({out.stat().st_size/1e6:.0f} MB)")
            return
        except Exception as e:
            got = out.stat().st_size if out.exists() else 0
            # Reset the stall counter whenever bytes advanced (progress).
            stalls = 0 if got > last_size else stalls + 1
            last_size = got
            if stalls > max_stalls:
                raise
            backoff = min(60, 2 ** stalls)   # 1,2,4,...,60s; waits out outages
            pct = (100 * got / expected_size) if expected_size else 0
            print(f"      [stall {stalls}/{max_stalls}] {type(e).__name__}: {e} "
                  f"| at {got/1e6:.0f} MB ({pct:.1f}%), backoff {backoff}s")
            time.sleep(backoff)


def download_zenodo_record(record_id: str, dest_dir: Path):
    """Download every file of a public Zenodo record (resumable)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    api = f"https://zenodo.org/api/records/{record_id}"
    print(f"  Querying {api}")
    with urllib.request.urlopen(api) as r:
        meta = json.loads(r.read().decode('utf-8'))

    files = meta.get('files', [])
    print(f"  {len(files)} file(s) in record {record_id}")
    for f in files:
        key = f.get('key') or f.get('filename')
        url = (f.get('links', {}) or {}).get('self') or f.get('links', {}).get('download')
        size = f.get('size', 0)
        out = dest_dir / key
        if size and out.exists() and out.stat().st_size >= size:
            print(f"    [skip] {key} already complete")
            continue
        print(f"    downloading {key} ({size/1e6:.1f} MB, resumable) ...")
        _download_resumable(url, out, expected_size=size or None)
    print(f"  Saved to {dest_dir}")


def download_kaggle_dataset(slug: str, dest_dir: Path):
    """Download + unzip a Kaggle dataset using the Kaggle API token.

    Auth (no password shared with anyone): create an API token at
    kaggle.com -> Account -> 'Create New API Token', save the downloaded
    kaggle.json to ~/.kaggle/kaggle.json (or %USERPROFILE%\\.kaggle\\).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        print("  [ERROR] The 'kaggle' package is not installed.\n"
              "    Install:  venv\\Scripts\\pip install kaggle\n"
              "    Auth:     place kaggle.json in %USERPROFILE%\\.kaggle\\ "
              "(Kaggle -> Account -> Create New API Token)")
        sys.exit(2)
    api = KaggleApi()
    api.authenticate()
    print(f"  Downloading Kaggle dataset '{slug}' -> {dest_dir}")
    api.dataset_download_files(slug, path=str(dest_dir), unzip=True, quiet=False)
    print(f"  Done. Extracted under {dest_dir}")


def extract_archives(src_dir: Path, out_dir: Path):
    """Extract every .zip found under src_dir into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = list(src_dir.rglob('*.zip'))
    if not zips:
        print(f"  [WARN] no .zip archives under {src_dir}")
    for z in zips:
        print(f"  extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out_dir)
    print(f"  Extracted into {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Normalise to folder-per-animal
# ─────────────────────────────────────────────────────────────────────────────

def _find_animal_level(root: Path):
    """Find directories whose immediate children are image files (animal dirs).

    Returns a list of (animal_id, animal_dir). Handles arbitrary nesting by
    scanning for the deepest directories that directly contain images.
    """
    animal_dirs = []
    for d in root.rglob('*'):
        if not d.is_dir():
            continue
        has_imgs = any(c.suffix.lower() in _IMG_EXT for c in d.iterdir() if c.is_file())
        if has_imgs:
            animal_dirs.append(d)
    return animal_dirs


def _find_label(img: Path, src_root: Path):
    """Locate a YOLO .txt label for an image across common layouts.

    Checks: (1) next to the image; (2) a sibling 'labels' dir (the standard
    YOLO 'images/'<->'labels/' split); (3) a top-level labels/ dir by stem.
    """
    cand = img.with_suffix('.txt')
    if cand.exists():
        return cand
    # images/ -> labels/ in the path
    parts = list(img.parts)
    if 'images' in parts:
        swapped = Path(*[('labels' if p == 'images' else p) for p in parts]).with_suffix('.txt')
        if swapped.exists():
            return swapped
    sib = img.parent.parent / 'labels' / f'{img.stem}.txt'
    if sib.exists():
        return sib
    top = src_root / 'labels' / f'{img.stem}.txt'
    if top.exists():
        return top
    return None


def _roi_lower_center(img):
    """Heuristic muzzle ROI for a frontal cow face (no bounding box available).

    In a frontal face the muzzle/nostrils sit in the lower-central region, so we
    crop the central ~70% width and the lower ~55% height. This is an
    approximation to eyeball, NOT a detector — inspect a sample before trusting
    it, and prefer YOLO boxes or a real detector when available.
    """
    W, H = img.size
    left, right = int(0.15 * W), int(0.85 * W)
    top, bottom = int(0.40 * H), int(0.95 * H)
    return img.crop((left, top, right, bottom))


def _yolo_crop(image_path: Path, label_path: Path, target_class=None):
    """Crop the (first / target-class) YOLO box from an image. Returns PIL.Image."""
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    W, H = img.size
    best = None
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if target_class is not None and cls != target_class:
            continue
        xc, yc, w, h = (float(parts[1]) * W, float(parts[2]) * H,
                        float(parts[3]) * W, float(parts[4]) * H)
        box = (max(0, xc - w / 2), max(0, yc - h / 2),
               min(W, xc + w / 2), min(H, yc + h / 2))
        area = (box[2] - box[0]) * (box[3] - box[1])
        if best is None or area > best[0]:
            best = (area, box)
    if best is None:
        return img  # no label -> use full image
    return img.crop(tuple(int(v) for v in best[1]))


def organize(src: Path, out: Path, crop_muzzle: bool, target_class,
             roi_fallback: str = 'none'):
    """Copy/crop images into out/<animal_id>/<image> layout."""
    out.mkdir(parents=True, exist_ok=True)
    animal_dirs = _find_animal_level(src)
    if not animal_dirs:
        print(f"  [ERROR] No image-containing directories under {src}")
        sys.exit(1)

    # Disambiguate animal ids by their path relative to src.
    n_imgs, n_cropped, n_nolabel = 0, 0, 0
    for d in animal_dirs:
        if d.name == 'labels':          # skip YOLO label dirs treated as "animals"
            continue
        rel = d.relative_to(src)
        # Drop non-identity path components so ids are just the animal name.
        _skip = {'images', 'image', 'labels', 'train', 'val', 'test', 'data'}
        parts = [p for p in rel.parts if p.lower() not in _skip]
        animal_id = "__".join(parts) if parts else d.name
        dst_dir = out / animal_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img in d.iterdir():
            if img.suffix.lower() not in _IMG_EXT:
                continue
            if crop_muzzle:
                from PIL import Image
                label = _find_label(img, src)
                if label is not None:
                    cropped = _yolo_crop(img, label, target_class)
                    n_cropped += 1
                else:
                    full = Image.open(img).convert('RGB')
                    cropped = _roi_lower_center(full) if roi_fallback == 'lower-center' else full
                    n_nolabel += 1
                cropped.save(dst_dir / f"{img.stem}.png")
            else:
                shutil.copy2(img, dst_dir / img.name)
            n_imgs += 1

    n_animals = len([p for p in out.iterdir() if p.is_dir()])
    print(f"  Organized {n_imgs} images into {n_animals} animal folders at {out}")
    if crop_muzzle:
        print(f"  Muzzle-cropped via YOLO labels: {n_cropped} | "
              f"no label (kept full image): {n_nolabel}")
        if n_nolabel and not n_cropped:
            if roi_fallback == 'lower-center':
                print("  [INFO] No YOLO labels found; used the lower-center ROI\n"
                      "         heuristic for full-face images. INSPECT a sample —\n"
                      "         it is an approximation, not a detector.")
            else:
                print("  [WARN] No YOLO labels found anywhere. Kept full images.\n"
                      "         If these are full faces, re-run with\n"
                      "         --roi-fallback lower-center (or use a detector).")
    print(f"  -> evaluate: python scripts/evaluate_cross_dataset.py --data-root {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zenodo-record', default='10535934',
                    help='Zenodo record id (default: Pakistan muzzle dataset).')
    ap.add_argument('--kaggle-dataset', default=None,
                    help="Kaggle dataset slug 'owner/name' to download instead of Zenodo.")
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--organize', action='store_true')
    ap.add_argument('--crop-muzzle', action='store_true',
                    help='Crop YOLO boxes (if .txt labels sit next to images).')
    ap.add_argument('--target-class', type=int, default=None,
                    help='YOLO class id of the muzzle (if multi-class labels).')
    ap.add_argument('--roi-fallback', choices=['none', 'lower-center'], default='none',
                    help="Crop when no YOLO label exists: 'lower-center' heuristic "
                         "for full-face images, or 'none' to keep the full image.")
    ap.add_argument('--src', default=None, help='Source dir for --organize/--extract.')
    ap.add_argument('--out', default=None, help='Output dir for --organize.')
    args = ap.parse_args()

    tag = args.kaggle_dataset.replace('/', '__') if args.kaggle_dataset else str(args.zenodo_record)
    base = PROJECT_ROOT / 'data' / 'external' / tag

    if args.download:
        print("\n[1/3] Download")
        if args.kaggle_dataset:
            download_kaggle_dataset(args.kaggle_dataset, base / 'extracted')
        else:
            download_zenodo_record(args.zenodo_record, base / 'raw')
    if args.extract:
        print("\n[2/3] Extract")
        extract_archives(Path(args.src) if args.src else base / 'raw',
                         base / 'extracted')
    if args.organize:
        print("\n[3/3] Organize")
        src = Path(args.src) if args.src else base / 'extracted'
        out = Path(args.out) if args.out else (PROJECT_ROOT / 'data' / 'external' /
                                               f'{args.zenodo_record}_organized')
        organize(src, out, args.crop_muzzle, args.target_class, args.roi_fallback)

    if not any([args.download, args.extract, args.organize]):
        ap.print_help()


if __name__ == '__main__':
    main()
