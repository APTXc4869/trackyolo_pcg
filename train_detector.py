# Train PCG-YOLO detector (Ultralytics backend, after patch_ultralytics.py).

from __future__ import annotations
import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--pretrained", type=str, default="")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="train")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    if args.pretrained:
        model.load(args.pretrained)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        epochs=args.epochs,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        close_mosaic=10,
        amp=True,
    )


if __name__ == "__main__":
    main()
