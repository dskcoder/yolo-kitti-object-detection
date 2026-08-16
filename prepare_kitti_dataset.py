"""
Prepare official KITTI Object Detection dataset for YOLO training.

What this script does:
1. Extract official archives from data/kitti_official/raw.
2. Convert KITTI label_2 annotations to YOLO txt annotations.
3. Create deterministic train/val split.
4. Copy images and labels into data/kitti/{images,labels}/{train,val}.
"""

from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image


VALID_CLASSES = [
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(VALID_CLASSES)}


def extract_if_needed(image_zip: Path, label_zip: Path, extract_root: Path) -> None:
    image_dir = extract_root / "training" / "image_2"
    label_dir = extract_root / "training" / "label_2"

    if image_dir.exists() and label_dir.exists():
        print("Extraction already present, skipping archive extraction.")
        return

    extract_root.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {image_zip.name} ...")
    with zipfile.ZipFile(image_zip, "r") as zf:
        zf.extractall(extract_root)

    print(f"Extracting {label_zip.name} ...")
    with zipfile.ZipFile(label_zip, "r") as zf:
        zf.extractall(extract_root)


def kitti_line_to_yolo(line: str, img_w: int, img_h: int) -> str | None:
    parts = line.strip().split()
    if len(parts) < 8:
        return None

    cls_name = parts[0]
    if cls_name not in CLASS_TO_ID:
        return None

    x1 = float(parts[4])
    y1 = float(parts[5])
    x2 = float(parts[6])
    y2 = float(parts[7])

    x1 = max(0.0, min(x1, img_w - 1.0))
    x2 = max(0.0, min(x2, img_w - 1.0))
    y1 = max(0.0, min(y1, img_h - 1.0))
    y2 = max(0.0, min(y2, img_h - 1.0))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h

    return f"{CLASS_TO_ID[cls_name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def convert_labels_for_split(
    ids: list[str],
    image_dir: Path,
    label_dir: Path,
    out_image_dir: Path,
    out_label_dir: Path,
) -> tuple[int, int]:
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    total_boxes = 0

    for stem in ids:
        src_img = image_dir / f"{stem}.png"
        src_lbl = label_dir / f"{stem}.txt"

        if not src_img.exists() or not src_lbl.exists():
            continue

        with Image.open(src_img) as im:
            w, h = im.size

        yolo_lines: list[str] = []
        for line in src_lbl.read_text(encoding="utf-8").splitlines():
            yolo_line = kitti_line_to_yolo(line, w, h)
            if yolo_line is not None:
                yolo_lines.append(yolo_line)

        shutil.copy2(src_img, out_image_dir / src_img.name)
        (out_label_dir / src_lbl.name).write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        copied += 1
        total_boxes += len(yolo_lines)

    return copied, total_boxes


def prepare_dataset(raw_root: Path, extract_root: Path, out_root: Path, val_ratio: float, seed: int) -> None:
    image_zip = raw_root / "data_object_image_2.zip"
    label_zip = raw_root / "data_object_label_2.zip"

    if not image_zip.exists() or not label_zip.exists():
        raise FileNotFoundError("Official KITTI zip files were not found in data/kitti_official/raw.")

    extract_if_needed(image_zip, label_zip, extract_root)

    image_dir = extract_root / "training" / "image_2"
    label_dir = extract_root / "training" / "label_2"

    all_ids = sorted(p.stem for p in image_dir.glob("*.png"))
    if not all_ids:
        raise RuntimeError("No images found after extraction.")

    random.seed(seed)
    random.shuffle(all_ids)
    val_count = max(1, int(len(all_ids) * val_ratio))
    val_ids = sorted(all_ids[:val_count])
    train_ids = sorted(all_ids[val_count:])

    if out_root.exists():
        shutil.rmtree(out_root)

    train_img_dir = out_root / "images" / "train"
    val_img_dir = out_root / "images" / "val"
    train_lbl_dir = out_root / "labels" / "train"
    val_lbl_dir = out_root / "labels" / "val"

    print(f"Preparing train split with {len(train_ids)} images...")
    train_images, train_boxes = convert_labels_for_split(train_ids, image_dir, label_dir, train_img_dir, train_lbl_dir)

    print(f"Preparing val split with {len(val_ids)} images...")
    val_images, val_boxes = convert_labels_for_split(val_ids, image_dir, label_dir, val_img_dir, val_lbl_dir)

    print("Done.")
    print(f"Train images: {train_images}, train boxes: {train_boxes}")
    print(f"Val images: {val_images}, val boxes: {val_boxes}")
    print(f"Output root: {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official KITTI dataset for YOLO training")
    parser.add_argument("--raw-root", type=Path, default=Path("data/kitti_official/raw"))
    parser.add_argument("--extract-root", type=Path, default=Path("data/kitti_official/extracted"))
    parser.add_argument("--out-root", type=Path, default=Path("data/kitti"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(args.raw_root, args.extract_root, args.out_root, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
