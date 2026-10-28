"""Validation-only operating points; test scores are reused, never optimized."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baseline.data import read_jsonl, write_json, write_jsonl
from baseline.metrics import binary_metrics, evaluate_rows, task_predictions
from baseline.text_probe import TextProbe
from baseline.train_text_probe import load_records, point_rows, predict_records, subset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", default="artifacts/text_probe_v2/frozen/records.jsonl")
    p.add_argument("--checkpoint", default="artifacts/text_probe_v2/model")
    p.add_argument("--output", default="artifacts/text_probe_v2/operating_points")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    rows = load_records(args.records)
    model, config = TextProbe.load(args.checkpoint, args.device, torch.bfloat16)
    val = subset(rows, "val", config["max_eval_tasks"])
    probs = predict_records(model, val)
    event_tasks = {r["task_id"] for r in val if r["supervision"]["target_hijacked"] == 1
                   and not r["supervision"]["after_violation"]}
    selected = [(r, pr) for r, pr in zip(val, probs) if r["supervision"]["task_label"] == 0 or
                (r["task_id"] in event_tasks and not r["supervision"]["after_violation"])]
    tasks = task_predictions(point_rows([r for r, _ in selected]), [pr[0] for _, pr in selected])
    truth = [r["label"] for r in tasks.values()]
    scores = [r["score"] for r in tasks.values()]
    thresholds = sorted(set(scores) | {float(np.nextafter(max(scores), np.inf))})
    curve = [{"threshold": t, **binary_metrics(truth, scores, t)} for t in thresholds]
    feasible = [r for r in curve if r["fpr"] < .05]
    chosen = min(feasible, key=lambda r: (r["fnr"], r["fpr"], -r["threshold"]))
    test = subset(rows, "test", config["max_eval_tasks"])
    saved = {r["id"]: r["probabilities"] for r in read_jsonl(Path(args.checkpoint) / "test_predictions.jsonl")}
    if set(saved) != {r["id"] for r in test}:
        raise ValueError("Saved test predictions differ from the fixed seeded task set")
    report = {"threshold_selection": "validation only; minimize FNR subject to FPR strictly under 5%",
              "threshold_tuned_on_test": False, "strict_recall_validation": config["validation"],
              "fpr5_validation": chosen, "validation_curve": curve,
              "strict_recall_test": evaluate_rows(point_rows(test), [saved[r["id"]][0] for r in test], config["threshold"]),
              "fpr5_test": evaluate_rows(point_rows(test), [saved[r["id"]][0] for r in test], chosen["threshold"])}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    write_jsonl(output / "validation_predictions.jsonl", [{"id": r["id"], "probabilities": pr} for r, pr in zip(val, probs)])
    write_jsonl(output / "validation_tasks.jsonl", [{"task_id": k, **v} for k, v in tasks.items()])
    print(json.dumps({"validation": chosen, "test": report["fpr5_test"]}), flush=True)


if __name__ == "__main__":
    main()
