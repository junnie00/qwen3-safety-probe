"""Generate native AgentLAB attacks with an explicitly chosen planner model."""

import argparse
import json
import os
from pathlib import Path

from baseline.official.agentlab_search import openai_client, search


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output")
    p.add_argument("--attacker", choices=["gpt-5.1", "mimo-v2.6-flash"], default="gpt-5.1")
    p.add_argument("--key-file", help="MiMo key file; alternatively set MIMO_API_KEY")
    p.add_argument("--suites", nargs="+", default=["banking"], choices=["banking", "workspace", "travel", "slack"])
    p.add_argument("--user-task-id", nargs="+", default=["user_task_0"])
    p.add_argument("--injection-task-id", nargs="+", help="Defaults to the first available native injection task in each suite")
    p.add_argument("--full", action="store_true")
    p.add_argument("--max-rewrites", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--gpu-memory-fraction", type=float, default=1.0)
    args = p.parse_args()
    api_key, client = None, None
    if args.attacker == "mimo-v2.6-flash":
        api_key = Path(args.key_file).read_text().strip() if args.key_file else os.environ.get("MIMO_API_KEY")
        if not api_key:
            p.error("MiMo requires --key-file or MIMO_API_KEY")
    else:
        client = openai_client()  # Fail before GPU allocation if credentials absent.
    import torch
    from baseline.collect import load_models
    from baseline.official import agentlab
    from baseline.official.backend import Backend
    torch.set_num_threads(2)
    if not 0 < args.gpu_memory_fraction <= 1:
        p.error("GPU memory fraction must be in (0, 1]")
    for index in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, index)
    tokenizer, draft, target = load_models("models/Qwen3-0.6B", "models/Qwen3-8B", "cuda:0", "cuda:1", "bfloat16")
    backend = Backend(tokenizer, draft, target, None, {"threshold": .5}, max_new_tokens=args.max_new_tokens)
    output = Path(args.output or f"artifacts/official_benchmarks/{'mimo' if api_key else 'gpt51'}_attacks.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(line) for line in output.open()] if output.exists() else []
    done = {(r["suite"], r["user_task"], r["injection_task"]) for r in previous}
    with output.open("a") as stream:
        for name in args.suites:
            native = agentlab.suite(name)
            for user in list(native.user_tasks) if args.full else args.user_task_id:
                for injection in list(native.injection_tasks) if args.full else args.injection_task_id or list(native.injection_tasks)[:1]:
                    if (name, user, injection) in done:
                        continue
                    entry = {"suite": name, "user_task": user, "injection_task": injection}
                    bundle = search(backend, entry, client, output.parent / f"search_traces_{args.attacker}" / name / f"{user}_{injection}",
                                    args.max_rewrites, model=args.attacker, api_key=api_key)
                    stream.write(json.dumps(bundle) + "\n"); stream.flush()
                    print(json.dumps({**entry, "attempts": bundle["search_attempts"],
                                      "target_hijacked": bundle["search_victim_hijacked"]}), flush=True)


if __name__ == "__main__":
    main()
