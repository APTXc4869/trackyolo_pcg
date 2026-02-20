# Wrapper around TrackEval for MOT metrics.

from __future__ import annotations
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", type=str, required=True)
    p.add_argument("--pred_dir", type=str, required=True)
    p.add_argument("--seq_name", type=str, default="seq")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from trackeval import datasets, metrics, evaluator
    except Exception as e:
        print("[ERROR] TrackEval not installed. Do:")
        print("  git clone https://github.com/JonathonLuiten/TrackEval.git third_party/TrackEval")
        print("  pip install -e third_party/TrackEval")
        raise

    gt = Path(args.gt_dir).resolve()
    pr = Path(args.pred_dir).resolve()

    eval_cfg = evaluator.Evaluator.get_default_eval_config()
    eval_cfg["USE_PARALLEL"] = False

    ds_cfg = datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg["GT_FOLDER"] = str(gt)
    ds_cfg["TRACKERS_FOLDER"] = str(pr)
    ds_cfg["SEQMAP_FILE"] = ""
    ds_cfg["SEQ_INFO"] = {args.seq_name: None}
    ds_cfg["TRACKERS_TO_EVAL"] = ["."]
    ds_cfg["TRACKER_SUB_FOLDER"] = ""
    ds_cfg["OUTPUT_SUB_FOLDER"] = "trackeval"

    dataset_list = [datasets.MotChallenge2DBox(ds_cfg)]
    metric_list = [metrics.HOTA(), metrics.CLEAR(), metrics.Identity()]
    evaluator.Evaluator(eval_cfg).evaluate(dataset_list, metric_list)


if __name__ == "__main__":
    main()
