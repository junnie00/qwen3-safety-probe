"""Run actual BFCL-v4, LiveCodeBench v6, and AgentLAB Task-Injection episodes.

The small preset is an integration pilot, not a full leaderboard score. Existing
BIPIA-trained Probe weights/threshold are frozen throughout this evaluation.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
from statistics import mean, stdev
from time import perf_counter
import traceback

from baseline.official import agentlab, bfcl, lcb
from baseline.official.backend import Backend
from baseline.official.native import ROOT


BFCL_CATEGORIES = ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel",
                   "parallel_multiple", "irrelevance", "live_simple", "live_multiple", "live_parallel",
                   "live_parallel_multiple", "live_irrelevance", "live_relevance", "multi_turn_base",
                   "multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param"]


def select(benchmarks, full=False, attacks=None):
    tasks = []
    if "bfcl" in benchmarks:
        for category in BFCL_CATEGORIES if full else ["simple_python", "multiple", "irrelevance", "multi_turn_base"]:
            entries, answers = bfcl.load(category)
            for entry in entries if full else entries[:1]:
                tasks.append({"benchmark": "BFCL-v4", "id": entry["id"], "entry": entry,
                              "answer": answers.get(entry["id"]), "truth": 0})
    if "livecodebench" in benchmarks:
        entries = lcb.load()
        # Fixed before generation or Probe scoring. Includes functional/stdin code
        # as present in the selected native questions; no test outcomes consulted.
        selected = entries if full else sorted([e for e in entries if e["difficulty"] == "easy"],
                                               key=lambda e: (e["contest_date"], e["question_id"]))[:2]
        for entry in selected:
            tasks.append({"benchmark": "LiveCodeBench-v6", "id": entry["question_id"], "entry": entry, "truth": 0})
    if "agentlab" in benchmarks:
        if attacks and not full:
            for name, user in sorted({(key[0], key[1]) for key in attacks}):
                native = agentlab.suite(name)
                if user not in native.user_tasks:
                    raise ValueError(f"Unknown native user task: {name}/{user}")
                tasks.append({"benchmark": "AgentLAB-Task-Injection", "id": f"{name}/{user}/clean",
                              "entry": {"suite": name, "user_task": user}, "truth": 0})
            for name, user, injection in sorted(attacks):
                if injection not in agentlab.suite(name).injection_tasks:
                    raise ValueError(f"Unknown native injection task: {name}/{injection}")
                tasks.append({"benchmark": "AgentLAB-Task-Injection", "id": f"{name}/{user}/{injection}",
                              "entry": {"suite": name, "user_task": user, "injection_task": injection}, "truth": None})
            return tasks
        for name in ["banking", "workspace", "travel", "slack"] if full else ["banking"]:
            native = agentlab.suite(name)
            users = list(native.user_tasks) if full else ["user_task_0"]
            injections = list(native.injection_tasks) if full else ["injection_task_0"]
            for user in users:
                tasks.append({"benchmark": "AgentLAB-Task-Injection", "id": f"{name}/{user}/clean",
                              "entry": {"suite": name, "user_task": user}, "truth": 0})
                for injection in injections:
                    tasks.append({"benchmark": "AgentLAB-Task-Injection", "id": f"{name}/{user}/{injection}",
                                  "entry": {"suite": name, "user_task": user, "injection_task": injection}, "truth": None})
    return tasks


def summarize(rows, settings):
    report = {"settings": settings, "scope": "official_subset" if not settings["full"] else "supported_categories_full",
              "guide_targets_met": False, "by_benchmark": {}, "errors": [], "blocked": []}
    for row in rows:
        if row["status"] != "completed":
            report["blocked" if row["status"] == "blocked" else "errors"].append({
                k: row.get(k) for k in ["benchmark", "id", "repeat", "mode", "reason"]})
    for benchmark in sorted({r["benchmark"] for r in rows}):
        selected = [r for r in rows if r["benchmark"] == benchmark and r["status"] == "completed"]
        controls = {(r["id"], r["repeat"]): r["truth"] for r in selected if r["mode"] == "unprotected"}
        def detection_truth(row):
            if row["mode"] == "intercept" and benchmark.startswith("AgentLAB") and not row["id"].endswith("/clean"):
                return controls.get((row["id"], row["repeat"]))
            return row["truth"]
        values = {}
        for mode in sorted({r["mode"] for r in selected}):
            rr = [r for r in selected if r["mode"] == mode]
            blocks = [b for r in rr for c in r["calls"] for b in c["blocks"]]
            negative = [r for r in rr if detection_truth(r) == 0]
            positive = [r for r in rr if detection_truth(r) == 1]
            attacked = [r for r in rr if benchmark.startswith("AgentLAB") and not r["id"].endswith("/clean")]
            ntokens = sum(len(c["output_ids"]) for r in rr for c in r["calls"])
            seconds = sum(c["request_ms"] for r in rr for c in r["calls"]) / 1000
            proposed = sum(b["proposed_draft_tokens"] for b in blocks)
            values[mode] = {"episodes": len(rr), "unique_tasks": len({r["id"] for r in rr}),
                "native_accuracy_or_pass_at_1": mean(r["result"]["native_valid"] for r in rr),
                "truncated_requests": sum(c["finish_reason"] == "length" for r in rr for c in r["calls"]),
                "negative_episodes": len(negative), "positive_episodes": len(positive),
                "task_false_alarm_rate": mean(r["alarm"] for r in negative) if negative and mode != "unprotected" else None,
                "negative_block_alarm_rate": mean(b["alarm"] for r in negative for c in r["calls"] for b in c["blocks"]) if negative and mode != "unprotected" and blocks else None,
                "task_miss_rate_any_alarm": mean(not r["alarm"] for r in positive) if positive and mode != "unprotected" else None,
                "timely_fnr": None,
                "asr": mean(r["truth"] for r in attacked) if attacked else None,
                "draft_acceptance_rate": sum(b["accepted_draft_tokens"] for b in blocks) / proposed if proposed else None,
                "mean_victim_request_ms_per_episode": seconds * 1000 / len(rr),
                "output_tokens": ntokens, "tokens_per_second": ntokens / seconds if seconds else None,
                "tokens_per_second_per_allocated_gpu": ntokens / seconds / len(set(settings["model_devices"])) if seconds else None,
                "mean_probe_wait_ms_per_block": mean(b["probe_wait_ms"] for b in blocks) if blocks else None,
                "intercepted_episodes": sum(r["result"]["intercepted"] for r in rr)}
        pairs = {}
        for row in selected:
            pairs.setdefault((row["id"], row["repeat"]), {})[row["mode"]] = row
        matched = [p for p in pairs.values() if "unprotected" in p and "silent" in p]
        comparison = {"paired_episodes": len(matched)}
        if matched:
            comparison["identical_silent_target_tokens"] = all(
                [c["output_ids"] for c in p["unprotected"]["calls"]] == [c["output_ids"] for c in p["silent"]["calls"]]
                for p in matched)
            control = sum(c["request_ms"] for p in matched for c in p["unprotected"]["calls"])
            silent = sum(c["request_ms"] for p in matched for c in p["silent"]["calls"])
            comparison["victim_request_latency_change_percent"] = (silent / control - 1) * 100
            repeated = []
            for repeat in sorted({p["silent"]["repeat"] for p in matched}):
                pp = [p for p in matched if p["silent"]["repeat"] == repeat]
                before = sum(c["request_ms"] for pair in pp for c in pair["unprotected"]["calls"])
                after = sum(c["request_ms"] for pair in pp for c in pair["silent"]["calls"])
                repeated.append((after / before - 1) * 100)
            comparison["latency_change_percent_by_repeat"] = repeated
            comparison["latency_change_percent_mean"] = mean(repeated)
            comparison["latency_change_percent_std"] = stdev(repeated) if len(repeated) > 1 else None
        report["by_benchmark"][benchmark] = {"by_mode": values, "comparison": comparison}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="artifacts/official_benchmarks/pilot")
    p.add_argument("--benchmarks", nargs="+", choices=["bfcl", "livecodebench", "agentlab"], default=["bfcl", "livecodebench", "agentlab"])
    p.add_argument("--full", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--modes", nargs="+", choices=["unprotected", "silent", "intercept"], default=["unprotected", "silent"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--probe-checkpoint", default="artifacts/text_probe_v2/model_fpr5_fused")
    p.add_argument("--agentlab-attacks", help="JSONL native injection bundles with the planner model declared")
    p.add_argument("--gpu-memory-fraction", type=float, default=1.0)
    p.add_argument("--draft-device", default="cuda:0")
    p.add_argument("--target-device", default="cuda:1")
    p.add_argument("--probe-device", default="cuda:0")
    args = p.parse_args()
    if args.repeats < 1 or args.max_new_tokens < 1:
        p.error("Positive repeat/token counts required")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    attacks = agentlab.load_attacks(args.agentlab_attacks) if args.agentlab_attacks else {}
    tasks = select(args.benchmarks, args.full, attacks)
    manifest = [{k: t[k] for k in ["benchmark", "id"]} for t in tasks]
    (output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"selected": len(tasks), "tasks": manifest}), flush=True)
    if args.prepare_only:
        return
    import torch
    from baseline.collect import load_models
    from baseline.text_probe import TextProbe, TimedTextProbe
    torch.set_num_threads(2)
    if not 0 < args.gpu_memory_fraction <= 1:
        p.error("GPU memory fraction must be in (0, 1]")
    for index in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, index)
    tokenizer, draft, target = load_models("models/Qwen3-0.6B", "models/Qwen3-8B",
                                         args.draft_device, args.target_device, "bfloat16")
    model, config = TextProbe.load(args.probe_checkpoint, args.probe_device, torch.bfloat16)
    backend = Backend(tokenizer, draft, target, TimedTextProbe(model), config,
                      block_size=args.block_size, max_new_tokens=args.max_new_tokens)
    settings = {**vars(args), "probe_threshold": config["threshold"],
                "attacker_models": sorted({r["attack_model"] for r in attacks.values()}),
                "attack_bundles_sha256": hashlib.sha256(Path(args.agentlab_attacks).read_bytes()).hexdigest() if args.agentlab_attacks else None,
                "verifier_fusion": config.get("verifier_fusion", False), "checkpoint_frozen": True,
                "checkpoint_config_sha256": hashlib.sha256((Path(args.probe_checkpoint) / "config.json").read_bytes()).hexdigest(),
                "source_manifest_sha256": hashlib.sha256((ROOT / "artifacts/official_benchmarks/source_manifest.json").read_bytes()).hexdigest(),
                "hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch": torch.__version__, "model_devices": [args.draft_device, args.target_device, args.probe_device],
                "warmup_excluded": True, "temperature": 0, "kv_cache": True,
                "current_target_in_probe_input": False, "latency_subtraction": False,
                "timing": "sum of real victim requests including tokenization, target verification, Probe wait; native checker time excluded"}
    existing = []
    settings_path = output / "settings.json"
    if settings_path.exists():
        previous_settings = json.loads(settings_path.read_text())
        previous_settings.setdefault("gpu_memory_fraction", 1.0)
        previous_settings.setdefault("attacker_models", [])
        previous_settings.setdefault("attack_bundles_sha256", None)
        if previous_settings != settings:
            raise ValueError("Cannot resume with different experiment settings; use another output directory")
        existing = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    done = {(r["benchmark"], r["id"], r["repeat"], r["mode"]) for r in existing if r["status"] == "completed"}
    for mode in ["unprotected", "silent"]:
        backend.reset(mode)
        backend.complete([{"role": "user", "content": "Say hello."}], max_tokens=8)
    with (output / "runs.jsonl").open("a") as stream:
        for repeat in range(args.repeats):
            for task in tasks:
                for mode in args.modes if repeat % 2 == 0 else reversed(args.modes):
                    if (task["benchmark"], task["id"], repeat, mode) in done:
                        continue
                    row = {"benchmark": task["benchmark"], "id": task["id"], "repeat": repeat, "mode": mode,
                           "truth": task["truth"]}
                    entry = task["entry"]
                    attack = attacks.get((entry.get("suite"), entry.get("user_task"), entry.get("injection_task")))
                    if task["benchmark"].startswith("AgentLAB") and "injection_task" in entry and attack is None:
                        row.update(status="blocked", reason="Awaiting declared native search attack bundles; not scored")
                    else:
                        backend.reset(mode, seed=42 + repeat)
                        start = perf_counter()
                        try:
                            if task["benchmark"] == "BFCL-v4":
                                result = bfcl.run(backend, entry, task["answer"], f"spec_{repeat}_{mode}")
                            elif task["benchmark"] == "LiveCodeBench-v6":
                                result = lcb.run(backend, entry, output / f"lcb_{repeat}_{mode}_{task['id']}")
                            else:
                                result = agentlab.run(backend, entry, attack)
                                if attack is not None:
                                    row["truth"] = int(result["target_hijacked"])
                            row.update(status="completed", result=result,
                                       wall_ms_including_native_check=(perf_counter() - start) * 1000,
                                       calls=deepcopy(backend.calls), alarm=any(b["alarm"] for c in backend.calls for b in c["blocks"]))
                        except Exception as error:
                            row.update(status="error", reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                    stream.write(json.dumps(row, ensure_ascii=False, default=lambda x: x.model_dump()) + "\n"); stream.flush()
                    existing.append(row)
                    print(json.dumps({k: row.get(k) for k in ["benchmark", "id", "repeat", "mode", "status", "reason"]}), flush=True)
                    report = summarize(existing, settings)
                    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
