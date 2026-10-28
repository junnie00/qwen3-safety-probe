"""Resumable BIPIA end-position collection, local weak labeling, train and test.

Only the 0.6B draft provides probe features. The frozen 8B is also reused as an
OFFLINE labeler. Its judgments require human audit, especially at low FNR.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

import torch

from baseline.collect import collect, load_models
from baseline.data import load_cache, read_jsonl, write_json, write_jsonl
from baseline.evaluate import evaluate
from baseline.label_bipia import JUDGE_VERSION, judge_batch
from baseline.train import train


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    write_json(temp, value)
    temp.replace(path)


def save_judgments(path, judgments):
    temp = path.with_suffix(".tmp")
    write_jsonl(temp, judgments.values())
    temp.replace(path)


def execute(args, status):
    root = Path(args.output)
    cases = read_jsonl(args.cases)
    config = {**vars(args), "label_source": "local_Qwen3-8B_semantic_judge",
              "judge_version": JUDGE_VERSION,
              "requires_human_audit": True, "kv_cache_generation": False,
              "kv_cache_offline_judge": True}
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume configuration differs; use a new output directory")
    write_json(config_path, config)
    status(phase="loading_models", total_cases=len(cases), completed_cases=0)
    tokenizer, draft, target = load_models(args.draft_model, args.target_model,
                                         "cuda:0", "cuda:1", "bfloat16")
    complete, controls = 0, {}
    cases_by_id = {case["id"]: case for case in cases}
    for offset in range(0, len(cases), args.chunk_size):
        batch = cases[offset:offset + args.chunk_size]
        folder = root / "shards" / f"{offset // args.chunk_size:05d}"
        folder.mkdir(parents=True, exist_ok=True)
        feature_path, run_path = folder / "features.pt", folder / "runs.jsonl"
        status(phase="generating", completed_cases=complete, current_chunk=offset // args.chunk_size)
        if not feature_path.exists() or not run_path.exists():
            cache, runs = collect(batch, tokenizer, draft, target, block_size=4,
                                  max_new_tokens=args.max_new_tokens)
            temp = folder / "features.tmp"
            torch.save(cache, temp)
            temp.replace(feature_path)
            temp = folder / "runs.tmp"
            write_jsonl(temp, runs)
            temp.replace(run_path)
        else:
            cache = torch.load(feature_path, map_location="cpu", weights_only=True)
        runs = read_jsonl(run_path)
        outputs = {run["id"]: run["output"] for run in runs}
        for run in runs:
            if cases_by_id[run["id"]]["attack_str"] is None:
                controls[run["id"]] = run["output"]
        by_case = defaultdict(list)
        for row in cache["records"]:
            by_case[row["case_id"]].append(row)
        judgment_path = folder / "judgments.jsonl"
        judgments = {r["case_id"]: r for r in read_jsonl(judgment_path)} if judgment_path.exists() else {}
        pending = []
        for case in batch:
            if case["id"] in judgments:
                continue
            if case["attack_str"] is None:
                judgments[case["id"]] = {"case_id": case["id"], "task_label": 0,
                                          "first_affected_block": None, "reason": "clean_control"}
            elif outputs[case["id"]].strip() == controls[case["control_id"]].strip():
                judgments[case["id"]] = {"case_id": case["id"], "task_label": 0,
                                          "first_affected_block": None,
                                          "reason": "counterfactual_control_match"}
            else:
                evidence_case = {**case, "target_response": outputs[case["id"]],
                                 "clean_response": controls[case["control_id"]]}
                pending.append((evidence_case, by_case[case["id"]]))
        save_judgments(judgment_path, judgments)
        status(phase="offline_weak_labeling", completed_cases=complete,
               current_chunk=offset // args.chunk_size, current_chunk_judged=len(judgments))
        for first in range(0, len(pending), args.judge_batch_size):
            items = pending[first:first + args.judge_batch_size]
            answers = judge_batch(target, tokenizer, items)
            for (case, _), answer in zip(items, answers):
                judgments[case["id"]] = {"case_id": case["id"], **answer}
            save_judgments(judgment_path, judgments)
            status(phase="offline_weak_labeling", completed_cases=complete,
                   current_chunk=offset // args.chunk_size, current_chunk_judged=len(judgments))
        complete += len(batch)
        counts = Counter(str(r["task_label"]) for r in judgments.values())
        print(f"CHUNK COMPLETE {complete}/{len(cases)}; local judge labels {dict(counts)}", flush=True)
        status(phase="collecting", completed_cases=complete, current_chunk=offset // args.chunk_size,
               last_chunk_labels=dict(counts))
    del draft, target
    torch.cuda.empty_cache()
    status(phase="merging", completed_cases=complete)
    tensors, records, coverage, review = [], [], Counter(), []
    split_for_case = {case["id"]: case["split"] for case in cases}
    coverage_by_split = defaultdict(Counter)
    for folder in sorted((root / "shards").iterdir()):
        cache = torch.load(folder / "features.pt", map_location="cpu", weights_only=True)
        judgments = {r["case_id"]: r for r in read_jsonl(folder / "judgments.jsonl")}
        for case_id, answer in judgments.items():
            category = "unknown" if answer["task_label"] is None else "labeled"
            coverage[category] += 1
            coverage_by_split[split_for_case[case_id]][category] += 1
            if answer["task_label"] is None:
                review.append(answer)
        indices, positions = [], defaultdict(int)
        for i, original in enumerate(cache["records"]):
            answer = judgments[original["case_id"]]
            block = positions[original["case_id"]]
            positions[original["case_id"]] += 1
            label = answer["task_label"]
            if label is None:
                continue
            first = answer["first_affected_block"]
            row = dict(original)
            row.update(task_label=label, label=int(label == 1 and block >= first),
                       after_violation=bool(label == 1 and block > first))
            indices.append(i)
            records.append(row)
        if indices:
            tensors.append(cache["features"][indices])
    merged = {k: cache[k] for k in ["schema_version", "feature_source", "draft_model", "target_model", "collection_mode"]}
    if not tensors:
        raise ValueError("No usable weak labels; inspect raw judge outputs")
    merged.update(features=torch.cat(tensors), records=records,
                  label_source=config["label_source"], requires_human_audit=True)
    merged_path = root / "features_weak_labeled.pt"
    torch.save(merged, merged_path)
    write_jsonl(root / "pending_review.jsonl", review)
    write_json(root / "coverage.json", {**dict(coverage), "total": len(cases),
                                        "by_split": {k: dict(v) for k, v in coverage_by_split.items()},
                                        "unknowns_excluded_not_safe": True,
                                        "requires_human_audit": True})
    data = load_cache(merged_path)
    status(phase="training_probe", completed_cases=complete, label_coverage=dict(coverage))
    train(data, root / "model", epochs=60, lr=0.001)
    status(phase="evaluating_weak_labeled_subset", completed_cases=complete)
    report = evaluate(data, root / "model/probe.pt", root / "evaluation")
    status(phase="complete", completed_cases=complete, label_coverage=dict(coverage),
           weak_label_test_metrics=report["task_metrics"], requires_human_audit=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="artifacts/bipia_email_end/cases.jsonl")
    parser.add_argument("--output", default="artifacts/bipia_email_end/run")
    parser.add_argument("--draft-model", default="models/Qwen3-0.6B")
    parser.add_argument("--target-model", default="models/Qwen3-8B")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    progress = {"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}

    def status(**fields):
        progress.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(root / "progress.json", progress)

    try:
        execute(args, status)
    except Exception as error:
        status(phase="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
