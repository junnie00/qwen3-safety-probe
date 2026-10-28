"""Evaluate all known three-class test checkpoints without refitting anything."""

import argparse
from collections import Counter

import torch

from baseline.data import write_json
from baseline.text_data import CLASS_NAMES
from baseline.text_probe import TextProbe
from baseline.train_text_probe import load_records, predict_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", default="artifacts/text_probe_v2/frozen/records.jsonl")
    p.add_argument("--annotations", default=None)
    p.add_argument("--checkpoint", default="artifacts/text_probe_v2/model")
    p.add_argument("--output", default="artifacts/text_probe_v2/model/classification_all_known.json")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    rows = [r for r in load_records(args.records, args.annotations) if r["split"] == "test"
            and r["supervision"]["class_id"] is not None and not r["supervision"]["after_violation"]]
    model, config = TextProbe.load(args.checkpoint, args.device, torch.bfloat16)
    if config["threshold"] is None:
        raise ValueError("Training and validation calibration have not finished")
    if set(config["train_group_ids"]) & {r["group_id"] for r in rows}:
        raise ValueError("Test groups overlap training")
    probabilities = predict_records(model, rows)
    matrix = [[0] * 3 for _ in range(3)]
    for row, probs in zip(rows, probabilities):
        matrix[row["supervision"]["class_id"]][max(range(3), key=lambda i: probs[i])] += 1
    report = {"class_names": CLASS_NAMES, "confusion_matrix": matrix,
              "counts": dict(Counter(r["supervision"]["class_id"] for r in rows)),
              "checkpoints": len(rows), "scope": "All known held-out checkpoint labels, not whole-task FNR/FPR",
              "requires_human_audit": True, "checkpoint_unchanged": True}
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
