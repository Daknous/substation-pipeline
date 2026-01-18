#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import csv

import pandas as pd
from tqdm import tqdm

from ultralytics import YOLO
import cv2

from src.helpers.osm_ids import parse_osm_id_digits


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--model_pt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_annotated_dir", default=None)
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip images already in the output CSV")
    ap.add_argument("--no-images", action="store_true",
                    help="Skip saving annotated images to save disk space")
    return ap.parse_args()


def main():
    args = parse_args()
    
    # Check environment variable as well for no-images mode
    save_images = not (args.no_images or os.getenv("PIPELINE_NO_IMAGES", "").lower() in ("true", "1", "yes"))

    images_dir = Path(args.images_dir)
    out_csv = Path(args.out_csv)
    summary_csv = Path(args.summary_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    # Only create annotated directory if we're saving images
    ann_dir = None
    if save_images and args.out_annotated_dir:
        ann_dir = Path(args.out_annotated_dir)
        ann_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results to determine what to skip
    processed_images = set()
    existing_detections = []
    existing_summaries = []
    
    if args.skip_existing:
        if out_csv.exists():
            try:
                existing_df = pd.read_csv(out_csv)
                processed_images = set(existing_df['image'].unique())
                existing_detections = existing_df.to_dict('records')
                print(f"⏭️  Skip mode: found {len(processed_images)} already-processed images in detections CSV")
            except Exception as e:
                print(f"⚠️ Could not load existing detections CSV: {e}")
        
        if summary_csv.exists():
            try:
                existing_summary_df = pd.read_csv(summary_csv)
                existing_summaries = existing_summary_df.to_dict('records')
            except Exception as e:
                print(f"⚠️ Could not load existing summary CSV: {e}")

    model = YOLO(args.model_pt)
    names = model.names  # class_id -> name

    img_paths = sorted(images_dir.glob(args.pattern))
    if not img_paths:
        raise SystemExit(f"❌ No images found in {images_dir} matching {args.pattern}")

    rows = existing_detections.copy() if args.skip_existing else []
    summary_rows = existing_summaries.copy() if args.skip_existing else []
    
    skipped_count = 0
    processed_count = 0

    for p in tqdm(img_paths, desc="YOLO inference"):
        # Skip if already processed
        if args.skip_existing and p.name in processed_images:
            skipped_count += 1
            continue
        
        stem = p.stem  # often digits-only
        osm_id = parse_osm_id_digits(stem)
        osm_id_str = str(osm_id) if osm_id is not None else ""

        results = model.predict(
            source=str(p),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        r = results[0]
        det_count = 0
        per_class = {}

        if r.boxes is not None and len(r.boxes) > 0:
            det_count = len(r.boxes)
            for b in r.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)

                cls_name = names.get(cls_id, str(cls_id))
                per_class[cls_name] = per_class.get(cls_name, 0) + 1

                rows.append({
                    "osm_id": osm_id_str,
                    "image": p.name,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "conf": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "w": w, "h": h,
                    "area_px": w * h,
                })

        # Only save annotated images if save_images is True
        if save_images and ann_dir:
            im = r.plot()  # numpy array (BGR)
            out_path = ann_dir / p.name
            cv2.imwrite(str(out_path), im)

        summary_rows.append({
            "osm_id": osm_id_str,
            "image": p.name,
            "n_detections": det_count,
            **{f"n_{k}": v for k, v in per_class.items()},
        })
        
        processed_count += 1

    # write detections csv
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
    else:
        # always produce the file
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "osm_id","image","class_id","class_name","conf","x1","y1","x2","y2","w","h","area_px"
            ])
            w.writeheader()

    # write summary csv
    pd.DataFrame(summary_rows).fillna(0).to_csv(summary_csv, index=False)

    print(f"✅ Complete: Processed={processed_count} | Skipped={skipped_count} | Total={len(img_paths)}")
    print(f"✅ wrote: {out_csv} ({len(rows)} total detections)")
    print(f"✅ wrote: {summary_csv} ({len(summary_rows)} images)")
    
    if save_images and ann_dir:
        print(f"✅ annotated: {ann_dir}")
    elif not save_images:
        print(f"📝 Annotated images were skipped (--no-images flag). Only CSV files were saved.")


if __name__ == "__main__":
    main()
    