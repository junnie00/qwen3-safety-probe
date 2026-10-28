"""Resume fresh KV-cached trajectories with exact generation/Probe contexts."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from baseline.collect import load_models
from baseline.data import read_jsonl, write_json
from baseline.speculative_cached import generate
from baseline.text_data import source_messages


def resolve_case(case, control=None):
    result = deepcopy(case)
    if case.get("multi_turn"):
        if control is None:
            raise ValueError("A follow-up needs its actual clean-control trajectory")
        initial = deepcopy(case["messages"])
        initial[-1]["content"] = case["clean_context"]
        messages = initial + [{"role": "assistant", "content": control["output"]},
                              {"role": "user", "content": "Check the updated retrieved material and answer again: " + case["question"]},
                              deepcopy(case["messages"][-1])]
        result["messages"] = messages
        result["context"] = {"ordered_messages": deepcopy(messages)}
        result["history_pairs"] = [{"draft": b["draft_text"], "target": b["target_new_text"],
                                   "accepted_draft_tokens": b["accepted_draft_tokens"],
                                   "proposed_draft_tokens": b["proposed_draft_tokens"]} for b in control["blocks"]]
    return result


def collect(args):
    torch.set_num_threads(2)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    config = {**vars(args), "prompt_format": "source-separated-v2", "kv_cache": True}
    path = root / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Resume configuration changed; choose a new collection directory")
    write_json(path, config)
    existing = read_jsonl(root / "runs.jsonl") if (root / "runs.jsonl").exists() else []
    done = {r["case_id"]: r for r in existing}
    skipped = {r["case_id"] for r in read_jsonl(root / "excluded.jsonl")} if (root / "excluded.jsonl").exists() else set()
    tokenizer, draft, target = load_models(args.draft_model, args.target_model,
                                         args.draft_device, args.target_device, "bfloat16")
    eos = target.config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos] if eos is not None else []
    started = perf_counter()
    with (root / "runs.jsonl").open("a", encoding="utf-8") as stream, (root / "excluded.jsonl").open("a") as excluded:
        for case in cases:
            if case["id"] in done or case["id"] in skipped:
                continue
            if case.get("multi_turn") and case["control_id"] not in done:
                excluded.write(json.dumps({"case_id": case["id"], "reason": "control_excluded"}) + "\n")
                skipped.add(case["id"])
                continue
            resolved = resolve_case(case, done.get(case["control_id"]))
            messages = source_messages(resolved["context"])
            assert messages == resolved["messages"], "Probe context differs from generation messages"
            prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     enable_thinking=False, tools=case.get("tools"))
            if len(prefix) > args.max_prompt_tokens:
                excluded.write(json.dumps({"case_id": case["id"], "reason": "prompt_exceeds_explicit_limit",
                                           "prompt_tokens": len(prefix), "limit": args.max_prompt_tokens}) + "\n")
                excluded.flush()
                skipped.add(case["id"])
                continue
            result = generate(draft, target, tokenizer, prefix, resolved["context"], block_size=args.block_size,
                              max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
                              history_pairs=resolved.get("history_pairs", []),
                              draft_temperature=case.get("draft_temperature", 0.))
            result.update(case_id=case["id"], case=resolved, prompt_tokens=len(prefix),
                          prompt_ids_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
                          generation_prompt="source-separated-v2", target_temperature=0.,
                          draft_temperature=case.get("draft_temperature", 0.),
                          naturally_finished=bool(result["output_ids"] and result["output_ids"][-1] in eos))
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            done[case["id"]] = result
            status = {"completed": len(done), "excluded": len(skipped), "total": len(cases),
                      "last_case": case["id"], "last_blocks": len(result["blocks"]),
                      "elapsed_seconds_this_process": perf_counter() - started}
            write_json(root / "progress.json", status)
            print(json.dumps(status), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", default="artifacts/text_probe_v2/cases.jsonl")
    p.add_argument("--output", default="artifacts/text_probe_v2/collection")
    p.add_argument("--draft-model", default="models/Qwen3-0.6B")
    p.add_argument("--target-model", default="models/Qwen3-8B")
    p.add_argument("--draft-device", default="cuda:0")
    p.add_argument("--target-device", default="cuda:1")
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0)
    collect(p.parse_args())


if __name__ == "__main__":
    main()
