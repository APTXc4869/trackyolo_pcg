# Inference for PCG-YOLO detector using Ultralytics.

from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--save_dir", type=str, default="runs/predict_pcg")
    p.add_argument("--save_json", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=True,
        save_json=args.save_json,
        project=str(out.parent),
        name=out.name,
        verbose=False,
    )
    print(f"[DONE] Results saved to: {out}")


if __name__ == "__main__":
    main()
