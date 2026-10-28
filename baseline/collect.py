"""Collect aligned draft features and target-verified text using local weights."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from baseline.data import goal_label, read_jsonl, write_json, write_jsonl
from baseline.probe import FEATURE_SOURCE, load_probe
from baseline.speculative import generate


def load_models(draft_path, target_path, draft_device, target_device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if draft_device.startswith("npu") or target_device.startswith("npu"):
        import torch_npu  # noqa: F401; registered only on an Ascend installation
    tokenizer = AutoTokenizer.from_pretrained(draft_path, local_files_only=True)
    target_tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    if tokenizer.get_vocab() != target_tokenizer.get_vocab():
        raise ValueError("Greedy speculative verification requires identical token-ID mappings")
    models = []
    for path, device in [(draft_path, draft_device), (target_path, target_device)]:
        model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, torch_dtype=getattr(torch, dtype))
        models.append(model.to(device).requires_grad_(False).eval())
    return tokenizer, *models


def collect(cases, tokenizer, draft, target, *, block_size, max_new_tokens,
            probe=None, threshold=0.5, mode="silent", capture_features=True):
    ids = [case["id"] for case in cases]
    if not ids:
        raise ValueError("The cases file is empty")
    if len(set(ids)) != len(ids):
        raise ValueError("Case IDs must be unique")
    features, records, runs = [], [], []
    for case in cases:
        request_start = perf_counter()
        prefix = tokenizer.apply_chat_template(
            case["messages"], tools=case.get("tools"), tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        max_context = min(draft.config.max_position_embeddings, target.config.max_position_embeddings)
        if len(prefix) + max_new_tokens > max_context:
            raise ValueError(f"{case['id']}: full history plus output exceeds model context; do not silently truncate")
        eos = target.config.eos_token_id
        eos = eos if isinstance(eos, list) else ([eos] if eos is not None else [])
        result = generate(draft, target, prefix, block_size=block_size,
                          max_new_tokens=max_new_tokens, eos_token_ids=eos,
                          probe=probe, threshold=threshold, intercept=mode == "intercept",
                          capture_features=capture_features)
        output = tokenizer.decode(result["output_ids"], skip_special_tokens=True)
        request_ms = (perf_counter() - request_start) * 1000
        rule = case.get("evaluator", {"kind": "manual"})
        task_label = case.get("task_label", goal_label(rule, output)) if mode == "silent" else None
        history, after_violation, case_rows = [], bool(case.get("after_violation", False)), []
        for block in result["blocks"]:
            history.extend(block.verified)
            text = tokenizer.decode(history, skip_special_tokens=True)
            label = goal_label(rule, text) if mode == "silent" else None
            supplied = case.get("block_labels", {})
            label = supplied.get(str(block.index), label) if mode == "silent" else None
            row = {
                "id": f"{case['id']}/block-{block.index}",
                "case_id": case["id"],
                "task_id": case.get("task_id", case["id"]),
                "group_id": case.get("group_id", case.get("task_id", case["id"])),
                "split": case.get("split", "test"),
                "label": label,
                "task_label": task_label,
                "after_violation": after_violation,
                "draft_text": tokenizer.decode(block.proposed, skip_special_tokens=True),
                "target_new_text": tokenizer.decode(block.verified, skip_special_tokens=True),
                "target_text_so_far": text,
                "accepted_draft_tokens": block.accepted,
                "proposed_draft_tokens": len(block.proposed),
                "score": block.score,
                "target_verify_ms": block.verify_ms,
                "probe_ms": block.probe_ms,
                "probe_wait_ms": block.probe_wait_ms,
                "probe_overlapped_target": block.probe_overlapped,
            }
            if block.feature is not None:
                features.append(block.feature)
            records.append(row)
            case_rows.append(row)
            after_violation = after_violation or label == 1
        runs.append({
            "id": case["id"], "task_id": case.get("task_id", case["id"]),
            "output": output, "task_label": task_label,
            "intercepted": result["intercepted"],
            "elapsed_ms": result["elapsed_ms"], "output_tokens": len(result["output_ids"]),
            "request_ms": request_ms,
            "draft_tokens": result["draft_tokens"], "accepted_tokens": result["accepted_tokens"],
            "target_agreed_tokens": result["target_agreed_tokens"],
            "acceptance_rate": result["accepted_tokens"] / result["draft_tokens"],
            "mean_accepted_length": result["accepted_tokens"] / len(case_rows),
        })
        print(f"{case['id']}: {len(case_rows)} blocks, {len(result['output_ids'])} output tokens, "
              f"target_hijacked={task_label}, elapsed_ms={result['elapsed_ms']:.2f}", flush=True)
    cache = {
        "schema_version": 1, "feature_source": FEATURE_SOURCE,
        "features": torch.stack(features) if features else torch.empty((0, draft.config.hidden_size)),
        "records": records,
        "draft_model": str(draft.config._name_or_path),
        "target_model": str(target.config._name_or_path),
        "collection_mode": mode,
    }
    return cache, runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--draft-model", required=True, help="Local Qwen3-0.6B checkpoint/cache")
    parser.add_argument("--target-model", required=True, help="Local Qwen3-8B checkpoint/cache")
    parser.add_argument("--draft-device", default="cpu")
    parser.add_argument("--target-device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--checkpoint", help="Optional trained probe; it runs on CPU in a worker")
    parser.add_argument("--mode", choices=["silent", "intercept", "unprotected"], default="silent")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.mode == "intercept" and not args.checkpoint:
        parser.error("intercept requires --checkpoint")
    if args.mode == "unprotected" and args.checkpoint:
        parser.error("unprotected cannot load a probe")
    tokenizer, draft, target = load_models(args.draft_model, args.target_model,
                                         args.draft_device, args.target_device, args.dtype)
    probe, checkpoint = load_probe(args.checkpoint) if args.checkpoint else (None, {})
    if probe is not None and checkpoint["hidden_size"] != draft.config.hidden_size:
        raise ValueError("Probe input dimension differs from the draft hidden size")
    if probe is not None and (checkpoint["draft_model"] != str(draft.config._name_or_path)
                              or checkpoint["target_model"] != str(target.config._name_or_path)):
        raise ValueError("Use the same draft/target checkpoints as feature collection and training")
    cache, runs = collect(read_jsonl(args.cases), tokenizer, draft, target,
                          block_size=args.block_size, max_new_tokens=args.max_new_tokens,
                          probe=probe, threshold=checkpoint.get("threshold", 0.5),
                          mode=args.mode, capture_features=args.mode != "unprotected")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode != "unprotected":
        torch.save(cache, out / "features.pt")
    write_jsonl(out / "annotations.jsonl", cache["records"])
    write_jsonl(out / "runs.jsonl", runs)
    total_ms = sum(run["request_ms"] for run in runs)
    total_tokens = sum(run["output_tokens"] for run in runs)
    total_draft = sum(run["draft_tokens"] for run in runs)
    total_accepted = sum(run["accepted_tokens"] for run in runs)
    upstream = Path(__file__).resolve().parents[1] / "third_party/SafeSpec.UPSTREAM.json"
    cuda_devices = {next(model.parameters()).device for model in (draft, target)
                    if next(model.parameters()).device.type == "cuda"}
    gpu_memory = {str(device): {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    } for device in cuda_devices}
    write_json(out / "collection.json", {**vars(args), "feature_source": FEATURE_SOURCE,
                                         "kv_cache": False, "sampling": "greedy",
                                         "asynchronous_probe": probe is not None,
                                         "simulated_latency": False,
                                         "torch_version": str(torch.__version__),
                                         "source_cases_sha256": hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(),
                                         "upstream": json.loads(upstream.read_text()),
                                         "gpu_memory": gpu_memory,
                                         "summary": {
                                             "requests": len(runs),
                                             "mean_request_ms": total_ms / len(runs),
                                             "output_tokens": total_tokens,
                                             "tokens_per_second": total_tokens * 1000 / total_ms,
                                             "acceptance_rate": total_accepted / total_draft,
                                             "mean_accepted_length": total_accepted / len(cache["records"]),
                                         }})


if __name__ == "__main__":
    main()
