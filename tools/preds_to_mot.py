# Convert tracker preds.json to MOTChallenge prediction file.

from __future__ import annotations
import argparse, json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_json", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seq_name", type=str, default="seq")
    return p.parse_args()


def main():
    args = parse_args()
    pred = json.loads(Path(args.pred_json).read_text(encoding="utf-8"))
    out = Path(args.out_dir) / args.seq_name
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in pred:
        frame = int(r["frame"])
        tid = int(r["track_id"])
        x1, y1, x2, y2 = r["bbox_xyxy"]
        rows.append([frame, tid, x1, y1, x2-x1, y2-y1, float(r.get("score", 1.0)), int(r.get("cls", -1)), -1, -1])

    rows.sort(key=lambda x: (x[0], x[1]))
    pred_path = out / "pred.txt"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"[DONE] {pred_path}")


if __name__ == "__main__":
    main()
