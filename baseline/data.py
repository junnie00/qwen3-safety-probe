"""Explicit checkpoint labels and task truth; no inference from attack presence."""

import json
from pathlib import Path

import torch

from baseline.probe import FEATURE_SOURCE


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def goal_label(rule, target_text):
    """Only controlled exact-marker tasks have an automatic evaluator.

    Ordinary attacks require manual/application-specific judgments. Merely
    containing an injection is not evidence that the target was hijacked.
    """
    kind = rule.get("kind", "manual")
    if kind == "clean":
        return 0
    if kind == "marker":
        if not rule.get("text"):
            raise ValueError("A marker evaluator needs a nonempty text")
        return int(rule["text"] in target_text)
    if kind != "manual":
        raise ValueError(f"Unknown evaluator: {kind}")
    return None


def load_cache(path, annotations=None):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["feature_source"] != FEATURE_SOURCE:
        raise ValueError("Expected features extracted from the draft model")
    records = data["records"]
    if len(records) != len(data["features"]):
        raise ValueError("Feature and record counts differ")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate checkpoint IDs")
    if annotations:
        edits = read_jsonl(annotations)
        by_id = {r["id"]: r for r in edits}
        if len(by_id) != len(edits) or set(by_id) != set(ids):
            raise ValueError("Annotations must contain every checkpoint ID exactly once")
        for row in records:
            edit = by_id[row["id"]]
            for key in ["label", "task_label", "after_violation"]:
                if key in edit:
                    row[key] = edit[key]
    # All turns, seeds, and variants of one original task stay on one side.
    splits = {}
    task_truth = {}
    for row in records:
        if not isinstance(row["after_violation"], bool):
            raise ValueError("after_violation must be a JSON boolean")
        split = row["split"]
        if split not in ("train", "val", "test"):
            raise ValueError("Each case needs split=train, val, or test")
        group = row["group_id"]
        if group in splits and splits[group] != split:
            raise ValueError(f"Data leakage: group {group!r} crosses splits")
        splits[group] = split
        for key in ["label", "task_label"]:
            if row[key] is not None and row[key] not in (0, 1):
                raise ValueError(f"{key} must be 0, 1, or null")
        task = row["task_id"]
        value = (split, row["task_label"])
        if task in task_truth and task_truth[task] != value:
            raise ValueError(f"Inconsistent final truth/split for task {task!r}")
        task_truth[task] = value
    return data


def indices_for(data, split, *, training=False, require_checkpoint_labels=True):
    indices = [i for i, row in enumerate(data["records"])
               if row["split"] == split and not (training and
                   (row["after_violation"] or row.get("exclude_checkpoint_training", False)))]
    if not indices:
        raise ValueError(f"No eligible {split} records")
    for i in indices:
        row = data["records"][i]
        if row["task_label"] is None or (require_checkpoint_labels and row["label"] is None):
            raise ValueError(f"Unannotated checkpoint {row['id']}; fill annotations.jsonl first")
    return indices
