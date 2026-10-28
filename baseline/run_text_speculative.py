"""Source-aware real-model decoding with paired-history text-probe inference."""

import argparse
import json
import os
from pathlib import Path
import socket
from statistics import mean, stdev
from time import perf_counter

import torch

from baseline.collect import load_models
from baseline.data import read_jsonl, write_json, write_jsonl
from baseline.speculative_cached import generate
from baseline.text_data import context_from_case, source_messages, trace_records
from baseline.text_probe import TextProbe, TimedTextProbe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--draft-model", default="models/Qwen3-0.6B")
    p.add_argument("--target-model", default="models/Qwen3-8B")
    p.add_argument("--probe-checkpoint")
    p.add_argument("--draft-device", default="cuda:0")
    p.add_argument("--target-device", default="cuda:0")
    p.add_argument("--probe-device", default="cuda:1")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--mode", choices=["unprotected", "silent", "intercept", "compare"], default="silent")
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--prompt-format", choices=["source-separated", "original"], default="source-separated")
    args = p.parse_args()
    torch.set_num_threads(2)
    if args.mode != "unprotected" and not args.probe_checkpoint:
        p.error("Silent/intercept mode needs a trained probe checkpoint")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, draft, target = load_models(args.draft_model, args.target_model,
                                         args.draft_device, args.target_device, args.dtype)
    config, probe = None, None
    if args.mode != "unprotected":
        model, config = TextProbe.load(args.probe_checkpoint, args.probe_device, getattr(torch, args.dtype))
        if config.get("threshold") is None:
            p.error("Probe has no validation-calibrated threshold")
        probe = TimedTextProbe(model)
    cases = read_jsonl(args.cases)
    if args.mode == "compare":
        context = context_from_case(cases[0])
        messages = source_messages(context) if args.prompt_format == "source-separated" else cases[0]["messages"]
        prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False, tools=cases[0].get("tools"))
        generate(draft, target, tokenizer, prefix, context, block_size=args.block_size,
                 max_new_tokens=min(8, args.max_new_tokens), probe=probe, intercept=False,
                 history_pairs=cases[0].get("history_pairs", []),
                 verifier_fusion=config.get("verifier_fusion", False))
    runs, records, recorded_cases = [], [], set()
    devices = {args.draft_device, args.target_device}
    if probe is not None:
        devices.add(args.probe_device)
    for device in devices:
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
    with (out / "runs.jsonl").open("w", encoding="utf-8") as stream:
        for repeat in range(args.repeats):
            for case in cases:
                modes = ["unprotected", "silent"] if args.mode == "compare" else [args.mode]
                if repeat % 2:
                    modes.reverse()
                for mode in modes:
                    start = perf_counter()
                    context = context_from_case(case)
                    messages = source_messages(context) if args.prompt_format == "source-separated" else case["messages"]
                    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                            enable_thinking=False, tools=case.get("tools"))
                    limit = min(draft.config.max_position_embeddings, target.config.max_position_embeddings)
                    if len(prefix) + args.max_new_tokens > limit:
                        raise ValueError("Complete generation context exceeds a backbone's token limit")
                    eos = target.config.eos_token_id
                    eos = eos if isinstance(eos, list) else [eos] if eos is not None else []
                    result = generate(draft, target, tokenizer, prefix, context, block_size=args.block_size,
                                      max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                                      seed=args.seed + repeat, eos_token_ids=eos, probe=probe if mode != "unprotected" else None,
                                      threshold=config["threshold"] if config else .5, intercept=mode == "intercept",
                                      history_pairs=case.get("history_pairs", []),
                                      verifier_fusion=config.get("verifier_fusion", False) if config else False)
                    result.update(case_id=case["id"], repeat=repeat, mode=mode,
                                  request_ms=(perf_counter() - start) * 1000)
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    runs.append(result)
                    if repeat == 0 and mode != "intercept" and case["id"] not in recorded_cases:
                        records.extend(trace_records(case, result["blocks"]))
                        recorded_cases.add(case["id"])
                    print(json.dumps({"case": case["id"], "repeat": repeat, "mode": mode,
                                      "elapsed_ms": result["request_ms"], "tokens": len(result["output_ids"]),
                                      "intercepted": result["intercepted"]}), flush=True)
    write_jsonl(out / "records.jsonl", records)
    blocks = [b for r in runs for b in r["blocks"]]
    report = {**vars(args), "kv_cache": True, "simulated_latency": False,
              "probe_class_names": config.get("class_names") if config else None,
              "probe_readout": config.get("readout", "pointer") if config else None,
              "probe_threshold": config["threshold"] if config else None,
              "threshold_policy": config.get("threshold_policy", "validation_recall_constraint") if config else None,
              "inference_lora_merged": bool(config),
              "verifier_fusion": config.get("verifier_fusion", False) if config else False,
              "runs": len(runs), "mean_elapsed_ms": sum(r["elapsed_ms"] for r in runs) / len(runs),
              "accepted_draft_tokens": sum(r["accepted_tokens"] for r in runs),
              "proposed_draft_tokens": sum(r["draft_tokens"] for r in runs),
              "probe_overlap_blocks": sum(b["probe_overlapped_target"] for b in blocks),
              "probe_finished_before_target_blocks": sum(b["probe_finished_before_target"] for b in blocks),
              "total_blocks": len(blocks), "mean_probe_wait_ms": sum(b["probe_wait_ms"] for b in blocks) / len(blocks),
              "historical_rejected_drafts_preserved": True, "current_target_in_probe_input": False}
    report["by_mode"] = {}
    report["allocated_model_devices"] = sorted(devices)
    report["hardware"] = {"hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "torch": torch.__version__, "device_names": {d: torch.cuda.get_device_name(d)
                              for d in devices if d.startswith("cuda")},
                          "max_allocated_memory_bytes": {d: torch.cuda.max_memory_allocated(d)
                              for d in devices if d.startswith("cuda")}}
    report["warmup_runs_excluded"] = int(args.mode == "compare")
    for mode in sorted({r["mode"] for r in runs}):
        selected = [r for r in runs if r["mode"] == mode]
        selected_blocks = [b for r in selected for b in r["blocks"]]
        times = [r["request_ms"] for r in selected]
        report["by_mode"][mode] = {"mean_request_ms": mean(times), "std_request_ms": stdev(times) if len(times) > 1 else 0,
            "acceptance_rate": sum(r["accepted_tokens"] for r in selected) / sum(r["draft_tokens"] for r in selected),
            "mean_probe_wait_ms": mean(b["probe_wait_ms"] for b in selected_blocks),
            "blocks": len(selected_blocks),
            "probe_overlap_blocks": sum(b["probe_overlapped_target"] for b in selected_blocks),
            "probe_finished_before_target_blocks": sum(b["probe_finished_before_target"] for b in selected_blocks),
            "tokens_per_second_per_allocated_device": sum(len(r["output_ids"]) for r in selected) /
                (sum(times) / 1000 * len(devices))}
    if args.mode == "compare":
        pairs = {}
        for row in runs:
            pairs.setdefault((row["case_id"], row["repeat"]), {})[row["mode"]] = row
        report["identical_silent_outputs"] = all(p["silent"]["output_ids"] == p["unprotected"]["output_ids"] for p in pairs.values())
        assert report["identical_silent_outputs"], "Silent Probe changed speculative generation"
        differences = [p["silent"]["request_ms"] - p["unprotected"]["request_ms"] for p in pairs.values()]
        report["paired_latency_difference_ms"] = {"mean": mean(differences), "std": stdev(differences) if len(differences) > 1 else 0}
    write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
