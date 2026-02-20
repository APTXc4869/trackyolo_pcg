# Run Ship-ByteTrack on a video using detections from Ultralytics YOLO.

from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from trackyolo.tracker import Detection, ShipByteTrack


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="runs/track_pcg")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", type=str, default="0")

    p.add_argument("--track_high", type=float, default=0.6)
    p.add_argument("--track_low", type=float, default=0.1)
    p.add_argument("--new_track", type=float, default=0.7)
    p.add_argument("--track_buffer", type=int, default=30)
    p.add_argument("--match_thresh", type=float, default=0.7)

    p.add_argument("--use_reid", type=int, default=0)  # placeholder, see tracker.py
    p.add_argument("--save_video", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(str(out / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = ShipByteTrack(
        track_high=args.track_high,
        track_low=args.track_low,
        new_track=args.new_track,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        use_reid=bool(args.use_reid),
    )

    mot_rows = []
    pred_json = []

    frame_idx = 0
    pbar = tqdm(total=nframes if nframes > 0 else None, desc="Tracking")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            clses = res.boxes.cls.cpu().numpy().astype(int)
            for b, s, c in zip(boxes, scores, clses):
                dets.append(Detection(xyxy=b.astype(np.float32), score=float(s), cls=int(c)))

        tracks = tracker.update(dets)

        if writer is not None:
            vis = frame.copy()
            for t in tracks:
                x1, y1, x2, y2 = t.to_xyxy().astype(int).tolist()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"ID{t.track_id}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            writer.write(vis)

        for t in tracks:
            x1, y1, x2, y2 = t.to_xyxy()
            mot_rows.append([frame_idx, t.track_id, float(x1), float(y1), float(x2-x1), float(y2-y1), float(t.score), int(t.cls), -1, -1])
            pred_json.append({"frame": frame_idx, "track_id": t.track_id, "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                              "score": float(t.score), "cls": int(t.cls)})

        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()

    mot_path = out / "preds_mot.txt"
    with mot_path.open("w", encoding="utf-8") as f:
        for r in mot_rows:
            f.write(",".join(map(str, r)) + "\n")

    json_path = out / "preds.json"
    json_path.write_text(json.dumps(pred_json, indent=2), encoding="utf-8")

    print(f"[DONE] MOT: {mot_path}")
    print(f"[DONE] JSON: {json_path}")
    if args.save_video:
        print(f"[DONE] Video: {out/'tracked.mp4'}")


if __name__ == "__main__":
    main()
