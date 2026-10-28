"""Reuse saved real traces, reconstructing source-aware CAUSAL observations."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import torch

from baseline.data import read_jsonl, write_json, write_jsonl
from baseline.text_data import trace_records


def prepare(cases_path, run_dir, output, annotate_count=1024):
    cases = {c["id"]: c for c in read_jsonl(cases_path)}
    root, out = Path(run_dir), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    counts, candidates = Counter(), defaultdict(list)
    with (out / "records.jsonl").open("w", encoding="utf-8") as stream:
        for folder in sorted((root / "shards").iterdir()):
            cache = torch.load(folder / "features.pt", map_location="cpu", weights_only=True)
            judgments = {j["case_id"]: j for j in read_jsonl(folder / "judgments.jsonl")}
            blocks = defaultdict(list)
            for row in cache["records"]:
                blocks[row["case_id"]].append(row)
            for key, rows in blocks.items():
                case = cases[key]
                for record in trace_records(case, rows, judgments[key]):
                    record["provenance"] = {"source": str(folder), "label_source": judgments[key].get("label_source"),
                                            "requires_human_audit": True,
                                            "generation_prompt": "legacy_BIPIA_prompt; sources reconstructed from builder metadata"}
                    record["oracle"] = {"injected_instruction": case.get("attack_str"),
                                        "reference_answer": case.get("reference_answer"),
                                        "attack_family": case.get("attack_name", "").rsplit("-", 1)[0]}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    s = record["supervision"]
                    counts[(record["split"], str(s["class_id"]))] += 1
                    if s["target_hijacked"] == 0 and s["class_id"] is None and not s["after_violation"]:
                        # Enrich rejected drafts for the difficult B class; this
                        # selection metadata is NEVER a Probe input.
                        b = rows[int(record["id"].rsplit("-", 1)[1])]
                        candidates[record["split"]].append((b["accepted_draft_tokens"] < b["proposed_draft_tokens"], record))
    rng, selected = random.Random(42), []
    quotas = {"train": int(annotate_count * .625), "val": int(annotate_count * .125)}
    quotas["test"] = annotate_count - sum(quotas.values())
    for split, quota in quotas.items():
        pool = candidates[split]
        rng.shuffle(pool)
        pool.sort(key=lambda p: not p[0])
        per_case = Counter()
        for _, record in pool:
            if sum(r["split"] == split for r in selected) >= quota:
                break
            if per_case[record["case_id"]] < 2:
                selected.append(record)
                per_case[record["case_id"]] += 1
    write_jsonl(out / "draft_label_requests.jsonl", selected)
    report = {"source_cases": cases_path, "source_run": str(root),
              "checkpoint_counts": {f"{a}/{b}": n for (a, b), n in counts.items()},
              "draft_labels_requested": len(selected), "seed": 42,
              "sampling": "rejected drafts first, max two per case; not a population class-prior estimate",
              "current_target_is_supervision_only": True, "sources_reconstructed_without_changing_saved_outputs": True}
    write_json(out / "preparation.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", default="artifacts/bipia_email_end/cases.jsonl")
    p.add_argument("--run", default="artifacts/bipia_email_end/mimo_run")
    p.add_argument("--output", default="artifacts/text_probe")
    p.add_argument("--annotate-count", type=int, default=1024)
    a = p.parse_args()
    print(json.dumps(prepare(a.cases, a.run, a.output, a.annotate_count), indent=2))


if __name__ == "__main__":
    main()
