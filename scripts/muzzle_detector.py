"""
Muzzle Detector: train + crop
=============================
Trains a YOLO muzzle detector on a single-class YOLO dataset (e.g. the
Roboflow 'sharifashik/cow-muzzle-dataset'), then uses it to detect and crop
the muzzle region from full-scene cattle photos so a muzzle-texture model can
consume them.

This unlocks datasets like the Kaggle 25-ID 'Cattle Muzzle - DB', whose images
are wide farm scenes (whole cow + background) rather than muzzle crops, and it
doubles as the deployment-time detection front-end for the paper.

Usage:
    # 1. train (a few minutes on GPU)
    python scripts/muzzle_detector.py train \
        --data "data/external/sharifashik__cow-muzzle-dataset/extracted/data.yaml" \
        --epochs 40

    # 2. crop a folder-per-animal dataset using the trained detector
    python scripts/muzzle_detector.py crop \
        --weights outputs/detector/muzzle/weights/best.pt \
        --src "data/external/kollabathulakaushik__id/extracted/Cattle Muzzle - DB/Original" \
        --out data/external/kaggle25_cropped --pad 0.08

    # 3. evaluate transfer on the cropped set
    python scripts/evaluate_cross_dataset.py --data-root data/external/kaggle25_cropped
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def train(args):
    """Fine-tune a YOLOv8n single-class ('muzzle') detector on args.data (a YOLO
    data.yaml) and save the best weights under outputs/detector/<name>/."""
    from ultralytics import YOLO
    model = YOLO(args.base)  # e.g. yolov8n.pt (downloads pretrained weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(PROJECT_ROOT / 'outputs/detector'),
        name=args.name,
        device=0 if _cuda() else 'cpu',
        exist_ok=True,
        verbose=True,
    )
    best = PROJECT_ROOT / 'outputs/detector' / args.name / 'weights' / 'best.pt'
    print(f"\n[OK] Trained detector -> {best}")


def crop(args):
    """Run the trained detector over a folder of wide farm-scene images and write
    the highest-confidence muzzle crop per image (EXIF-aware, degenerate-box guard)
    — the deployment front-end and the external-dataset preprocessor."""
    from ultralytics import YOLO
    from PIL import Image, ImageOps
    model = YOLO(args.weights)

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Discover per-animal directories (immediate dirs containing images).
    animal_dirs = [d for d in sorted(src.iterdir()) if d.is_dir()]
    if not animal_dirs:
        animal_dirs = [src]

    n_img, n_det, n_miss = 0, 0, 0
    for d in animal_dirs:
        imgs = [p for p in sorted(d.iterdir()) if p.suffix.lower() in _IMG_EXT]
        if not imgs:
            continue
        dst = out / d.name
        dst.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(imgs), args.batch):
            batch = imgs[i:i + args.batch]
            results = model([str(p) for p in batch], verbose=False,
                            conf=args.conf, device=0 if _cuda() else 'cpu')
            for p, r in zip(batch, results):
                n_img += 1
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    n_miss += 1
                    continue
                # highest-confidence muzzle box
                conf = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()
                bi = conf.argmax()
                x1, y1, x2, y2 = xyxy[bi]
                # Match ultralytics' EXIF handling so box coords align with pixels.
                img = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
                W, H = img.size
                pw, ph = (x2 - x1) * args.pad, (y2 - y1) * args.pad
                left, upper = max(0, x1 - pw), max(0, y1 - ph)
                right, lower = min(W, x2 + pw), min(H, y2 + ph)
                if right - left < 2 or lower - upper < 2:   # degenerate box
                    n_miss += 1
                    continue
                img.crop((int(left), int(upper), int(right), int(lower))).save(dst / f"{p.stem}.png")
                n_det += 1
        print(f"  {d.name}: {len([p for p in dst.iterdir()])} crops")

    print(f"\n[OK] Cropped {n_det}/{n_img} images (missed {n_miss}) -> {out}")
    print(f"  -> python scripts/evaluate_cross_dataset.py --data-root {out}")


def _cuda():
    """True if a CUDA device is available (selects GPU vs CPU for YOLO)."""
    import torch
    return torch.cuda.is_available()


def main():
    """CLI entry point: `train` fine-tunes the detector, `crop` runs inference to
    extract muzzle crops."""
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train')
    t.add_argument('--data', required=True, help='YOLO data.yaml')
    t.add_argument('--base', default='yolov8n.pt')
    t.add_argument('--epochs', type=int, default=40)
    t.add_argument('--imgsz', type=int, default=640)
    t.add_argument('--batch', type=int, default=16)
    t.add_argument('--name', default='muzzle')
    t.set_defaults(func=train)

    c = sub.add_parser('crop')
    c.add_argument('--weights', required=True)
    c.add_argument('--src', required=True, help='folder-per-animal source images')
    c.add_argument('--out', required=True)
    c.add_argument('--conf', type=float, default=0.25)
    c.add_argument('--pad', type=float, default=0.08, help='padding as frac of box size')
    c.add_argument('--batch', type=int, default=16)
    c.set_defaults(func=crop)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
