# Convert COCO-style tracking GT json to MOTChallenge format.

from __future__ import annotations
import argparse, json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seq_name", type=str, default="seq")
    p.add_argument("--img_width", type=int, default=1920)
    p.add_argument("--img_height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=25)
    return p.parse_args()


def main():
    args = parse_args()
    coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
    images = coco.get("images", [])
    if any("frame_id" in im for im in images):
        img2frame = {im["id"]: int(im["frame_id"]) for im in images}
        seq_len = max(img2frame.values()) if img2frame else 0
    else:
        images_sorted = sorted(images, key=lambda x: x.get("file_name", ""))
        img2frame = {im["id"]: i + 1 for i, im in enumerate(images_sorted)}
        seq_len = len(images_sorted)

    out = Path(args.out_dir) / args.seq_name
    gt_dir = out / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for ann in coco.get("annotations", []):
        frame = img2frame.get(ann["image_id"])
        if frame is None:
            continue
        tid = ann.get("track_id", ann.get("id", 0))
        x, y, w, h = ann["bbox"]
        cls = ann.get("category_id", -1)
        lines.append([frame, tid, x, y, w, h, 1, cls, -1, -1])

    lines.sort(key=lambda r: (r[0], r[1]))

    gt_path = gt_dir / "gt.txt"
    with gt_path.open("w", encoding="utf-8") as f:
        for r in lines:
            f.write(",".join(map(str, r)) + "\n")

    seqinfo = f"""[Sequence]
name={args.seq_name}
imDir=img1
frameRate={args.fps}
seqLength={seq_len}
imWidth={args.img_width}
imHeight={args.img_height}
imExt=.jpg
"""
    (out / "seqinfo.ini").write_text(seqinfo, encoding="utf-8")
    print(f"[DONE] {gt_path}")


if __name__ == "__main__":
    main()
