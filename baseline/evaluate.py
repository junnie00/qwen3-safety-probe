"""Evaluate a frozen checkpoint threshold; never tune it on the test split."""

import argparse
from pathlib import Path

import torch

from baseline.data import indices_for, load_cache, write_json, write_jsonl
from baseline.metrics import evaluate_rows
from baseline.probe import load_probe


def evaluate(cache, checkpoint_path, output, split="test"):
    probe, checkpoint = load_probe(checkpoint_path)
    indices = indices_for(cache, split, require_checkpoint_labels=False)
    if cache.get("collection_mode", "silent") != "silent":
        raise ValueError("FNR/FPR require final truth from a complete silent run")
    if cache["draft_model"] != checkpoint["draft_model"] or cache["target_model"] != checkpoint["target_model"]:
        raise ValueError("Collection model IDs differ from training; do not silently mix backbones")
    rows = [cache["records"][i] for i in indices]
    if split == "test":
        used_groups = set(checkpoint["train_group_ids"]) | set(checkpoint["val_group_ids"])
        if any(r["group_id"] in used_groups for r in rows):
            raise ValueError("Test groups overlap training or threshold calibration")
    with torch.inference_mode():
        scores = probe(cache["features"][indices].float()).squeeze(-1).sigmoid().tolist()
    report = evaluate_rows(rows, scores, checkpoint["threshold"])
    report.update({"threshold": checkpoint["threshold"], "split": split,
                   "threshold_tuned_on_test": False, "synthetic": bool(cache.get("synthetic", False)),
                   "label_source": cache.get("label_source", "provided_checkpoint_annotations"),
                   "requires_human_audit": bool(cache.get("requires_human_audit", False)),
                   "note": "Point estimates; a small zero-miss test does not establish FNR <0.1%."})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", report)
    write_jsonl(out / "predictions.jsonl", [
        {"id": row["id"], "task_id": row["task_id"], "label": row["label"],
         "task_label": row["task_label"], "score": score,
         "alarm": score >= checkpoint["threshold"]}
        for row, score in zip(rows, scores)])
    print(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--annotations")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    evaluate(load_cache(args.features, args.annotations), args.checkpoint, args.output, args.split)


if __name__ == "__main__":
    main()
