"""Export completed collection shards as model-independent text records.

Only messages, prior accepted text and the current draft are probe inputs.
Current/future target outputs and labels are kept under supervision, never input.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from baseline.data import read_jsonl, write_json, write_jsonl


def text_case(case, rows, run, judgment, draft_model, target_model):
    steps, prefix = [], ""
    label, first = judgment["task_label"], judgment["first_affected_block"]
    for i, row in enumerate(rows):
        unknown = label is None or (label == 1 and first is None)
        checkpoint_label = None if unknown else int(label == 1 and i >= first)
        after = None if unknown else bool(label == 1 and i > first)
        steps.append({
            "id": row["id"],
            "input": {"accepted_prefix_text": prefix, "draft_text": row["draft_text"]},
            "supervision": {
                "target_new_text": row["target_new_text"],
                "target_text_so_far": row["target_text_so_far"],
                "accepted_draft_tokens": row["accepted_draft_tokens"],
                "proposed_draft_tokens": row["proposed_draft_tokens"],
                "checkpoint_label": checkpoint_label, "after_violation": after,
            },
        })
        prefix = row["target_text_so_far"]
    return {
        "case_id": case["id"], "group_id": rows[0]["group_id"], "split": rows[0]["split"],
        "input": {"messages": case["messages"]}, "draft_steps": steps,
        "supervision": {
            "target_response": run["output"], "task_label": label,
            "first_affected_block": first,
            "judgment": {k: v for k, v in judgment.items() if k != "raw"},
        },
        "metadata": {
            "draft_model": draft_model, "target_model": target_model,
            "attack_name": case.get("attack_name"),
            "attack_str": case.get("attack_str"),
            "reference_answer": case.get("reference_answer"),
            "label_source": judgment.get("label_source", "local_Qwen3-8B_semantic_judge"),
            "requires_human_audit": True,
            "text_representation": "decoded draft blocks; not a lossless token-ID archive",
        },
    }


def export_once(cases, run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    destination = output / "shards"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    index = json.loads(manifest_path.read_text())["shards"] if manifest_path.exists() else {}
    for folder in sorted((run_dir / "shards").iterdir()):
        sources = [folder / name for name in ["features.pt", "runs.jsonl", "judgments.jsonl"]]
        if not all(p.exists() for p in sources):
            continue
        target = destination / (folder.name + ".jsonl")
        if target.exists() and target.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in sources):
            continue
        runs = read_jsonl(sources[1])
        judgments = {r["case_id"]: r for r in read_jsonl(sources[2])}
        if set(judgments) != {r["id"] for r in runs}:
            continue  # The worker is still labeling this shard.
        cache = torch.load(sources[0], map_location="cpu", weights_only=True)
        records = defaultdict(list)
        for row in cache["records"]:
            records[row["case_id"]].append(row)
        exported = [text_case(cases[run["id"]], records[run["id"]], run, judgments[run["id"]],
                              cache["draft_model"], cache["target_model"]) for run in runs]
        temp = target.with_suffix(".tmp")
        write_jsonl(temp, exported)
        temp.replace(target)
        index[folder.name] = {"cases": len(exported), "checkpoints": len(cache["records"])}
    manifest = {
        "schema_version": 1, "format": "text-only inputs with separate supervision",
        "source_run_dir": str(run_dir),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "exported_cases": sum(v["cases"] for v in index.values()),
        "exported_checkpoints": sum(v["checkpoints"] for v in index.values()),
        "shards": index, "requires_human_audit": True,
        "probe_input_fields": ["input.messages", "draft_steps[].input.accepted_prefix_text",
                               "draft_steps[].input.draft_text"],
        "warning": "Do not use supervision/metadata as online probe inputs. Text alone does not guarantee cross-model transfer.",
    }
    temp = manifest_path.with_suffix(".tmp")
    write_json(temp, manifest)
    temp.replace(manifest_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="artifacts/bipia_email_end/cases.jsonl")
    parser.add_argument("--run-dir", default="artifacts/bipia_email_end/run")
    parser.add_argument("--output", default="artifacts/bipia_email_end/text")
    parser.add_argument("--watch", action="store_true", help="Sync new completed shards every 30 seconds")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    cases = {case["id"]: case for case in read_jsonl(args.cases)}
    while True:
        manifest = export_once(cases, args.run_dir, args.output)
        print(f"Text export: {manifest['exported_cases']} cases, "
              f"{manifest['exported_checkpoints']} checkpoints", flush=True)
        if not args.watch:
            break
        progress = Path(args.run_dir) / "progress.json"
        if progress.exists():
            status = json.loads(progress.read_text())
            if status["phase"] in ["complete", "failed"]:
                break
            proc = Path(f"/proc/{status['pid']}/cmdline")
            if not proc.exists() or b"baseline.run_bipia" not in proc.read_bytes():
                break
        time.sleep(30)


if __name__ == "__main__":
    main()
