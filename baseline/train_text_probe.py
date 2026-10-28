"""Train a separate Qwen3 LoRA text probe and freeze its validation threshold."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from time import perf_counter

import torch

from baseline.data import read_jsonl, write_json, write_jsonl
from baseline.metrics import calibrate, evaluate_rows, task_predictions
from baseline.text_data import CLASS_NAMES, class_id
from baseline.text_probe import TextProbe, encode


def load_records(path, annotations=None):
    records = read_jsonl(path)
    edits = {r["id"]: r for r in read_jsonl(annotations)} if annotations else {}
    formats = {r.get("provenance", {}).get("generation_prompt") for r in records}
    if "source-separated-v2" in formats and formats != {"source-separated-v2"}:
        raise ValueError("Mixed legacy and source-separated v2 trajectories")
    groups, ids = {}, set()
    for row in records:
        if row["id"] in ids:
            raise ValueError("Duplicate checkpoint")
        ids.add(row["id"])
        group, split = row["group_id"], row["split"]
        if group in groups and groups[group] != split:
            raise ValueError("A source group crosses data splits")
        groups[group] = split
        s = row["supervision"]
        if row["id"] in edits:
            s["draft_hijacked"] = edits[row["id"]]["draft_hijacked"]
            s["class_id"] = class_id(s["draft_hijacked"], s["target_hijacked"])
    if set(edits) - ids:
        raise ValueError("Draft annotations name unknown checkpoint IDs")
    return records


def point_rows(records):
    return [{"task_id": r["task_id"], "task_label": r["supervision"]["task_label"],
             "label": r["supervision"]["target_hijacked"],
             "after_violation": r["supervision"]["after_violation"]} for r in records]


def subset(records, split, max_tasks=None):
    records = [r for r in records if r["split"] == split and r["supervision"]["task_label"] is not None]
    if max_tasks:
        tasks = sorted({r["task_id"] for r in records})
        random.Random(42).shuffle(tasks)
        chosen = set(tasks[:max_tasks])
        records = [r for r in records if r["task_id"] in chosen]
    return records


@torch.inference_mode()
def predict_records(model, records):
    model.eval()
    probabilities, last = [], None
    for row in records:
        if row["task_id"] != last:
            model.reset_history()
        probabilities.append(model.predict(row["input"])["probabilities"])
        last = row["task_id"]
    return probabilities


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    rows = load_records(args.records, args.annotations)
    model = TextProbe.from_base(args.base, args.device, torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
                                args.rank, args.pointer_size, args.readout)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    eligible, dropped = {"train": [], "val": []}, []
    for row in rows:
        split, s = row["split"], row["supervision"]
        if split not in eligible or s["class_id"] is None or s["after_violation"]:
            continue
        try:
            encoded = encode(model.tokenizer, row["input"], args.max_train_tokens, model.readout)
        except ValueError as error:
            dropped.append({"id": row["id"], "reason": str(error)})
            continue
        eligible[split].append((row, encoded))
    counts = Counter(r["supervision"]["class_id"] for r, _ in eligible["train"])
    if set(counts) != {0, 1, 2}:
        raise ValueError(f"Need genuine labels for all three classes; observed {dict(counts)}")
    if not eligible["val"]:
        raise ValueError("No validation checkpoint labels")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "excluded_long_inputs.jsonl", dropped)
    params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    # Balance TRAIN draws, not validation/test. Oversampling does not create new
    # independent examples; report the genuine counts separately below.
    sampling_weights = torch.tensor([1 / counts[r["supervision"]["class_id"]]
                                     for r, _ in eligible["train"]], dtype=torch.double)
    best, history = float("inf"), []
    started = perf_counter()
    for epoch in range(args.epochs):
        model.train()
        rng = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.multinomial(sampling_weights, len(eligible["train"]), replacement=True, generator=rng).tolist()
        batches = []
        for start in range(0, len(order), args.batch_size * 16):
            window = sorted(order[start:start + args.batch_size * 16], key=lambda i: len(eligible["train"][i][1]["ids"]))
            batches.extend(window[i:i + args.batch_size] for i in range(0, len(window), args.batch_size))
        random.Random(args.seed + epoch).shuffle(batches)
        total = seen = 0
        for batch_index, indices in enumerate(batches):
            batch = [eligible["train"][i] for i in indices]
            truth = torch.tensor([r["supervision"]["class_id"] for r, _ in batch], device=model.device)
            optimizer.zero_grad()
            logits = model([e for _, e in batch])
            loss = torch.nn.functional.cross_entropy(logits, truth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            seen += len(batch)
            if batch_index % 10 == 0:
                print(json.dumps({"epoch": epoch + 1, "seen": seen, "total": len(order), "loss": total / seen}), flush=True)
        model.eval()
        val_total = 0
        with torch.inference_mode():
            for offset in range(0, len(eligible["val"]), args.batch_size):
                batch = eligible["val"][offset:offset + args.batch_size]
                truth = torch.tensor([r["supervision"]["class_id"] for r, _ in batch], device=model.device)
                val_total += float(torch.nn.functional.cross_entropy(model([e for _, e in batch]), truth, reduction="sum"))
        val_loss = val_total / len(eligible["val"])
        history.append({"epoch": epoch + 1, "train_loss": total / seen, "val_loss": val_loss,
                        "sampled_class_counts": dict(Counter(eligible["train"][i][0]["supervision"]["class_id"] for i in order))})
        print(json.dumps(history[-1]), flush=True)
        if val_loss < best:
            best = val_loss
            model.save(out, args.base, args.rank, args.pointer_size, threshold=None,
                       train_group_ids=sorted({r["group_id"] for r, _ in eligible["train"]}),
                       requires_human_audit=True, training_label_counts=dict(counts))
    elapsed = perf_counter() - started
    del model, optimizer, params
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    model, config = TextProbe.load(out, args.device, torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)
    val = subset(rows, "val", args.max_eval_tasks)
    probabilities = predict_records(model, val)
    event_tasks = {r["task_id"] for r in val if r["supervision"]["target_hijacked"] == 1
                   and not r["supervision"]["after_violation"]}
    calibration_pairs = [(r, p) for r, p in zip(val, probabilities)
                         if r["supervision"]["task_label"] == 0 or
                         (r["task_id"] in event_tasks and not r["supervision"]["after_violation"])]
    tasks = task_predictions(point_rows([r for r, _ in calibration_pairs]), [p[0] for _, p in calibration_pairs])
    calibration = calibrate([t["label"] for t in tasks.values()], [t["score"] for t in tasks.values()])
    config.update(threshold=calibration["threshold"], validation=calibration,
                  validation_task_ids=sorted(tasks), max_eval_tasks=args.max_eval_tasks,
                  calibration_scope="positive trajectories through first event; negative trajectories complete")
    write_json(out / "config.json", config)
    write_json(out / "training.json", {"history": history, "training_seconds": elapsed,
               "train_class_counts": dict(counts), "val_class_counts": dict(Counter(r["supervision"]["class_id"] for r, _ in eligible["val"])),
               "trainable_parameters": trainable_count, "validation": calibration,
               "max_train_tokens": args.max_train_tokens, "excluded_long_inputs": len(dropped),
               "seed": args.seed, "epochs": args.epochs, "base": args.base, "readout": args.readout,
               "batch_size": args.batch_size, "lr": args.lr, "dtype": args.dtype, "rank": args.rank,
               "sampling": "class-balanced train draws with replacement; counts are original unique checkpoints",
               "batching": "sort within 16-batch windows by token length, then shuffle batches",
               "requires_human_audit": True})
    test = subset(rows, "test", args.max_eval_tasks)
    test_probs = predict_records(model, test)
    report = evaluate_rows(point_rows(test), [p[0] for p in test_probs], calibration["threshold"])
    known = [(r, p) for r, p in zip(test, test_probs) if r["supervision"]["class_id"] is not None and not r["supervision"]["after_violation"]]
    matrix = [[0] * 3 for _ in range(3)]
    for row, p in known:
        matrix[row["supervision"]["class_id"]][max(range(3), key=lambda i: p[i])] += 1
    report.update(class_names=CLASS_NAMES, class_confusion_matrix=matrix,
                  test_tasks=len({r["task_id"] for r in test}), threshold=calibration["threshold"],
                  threshold_tuned_on_test=False, max_eval_tasks=args.max_eval_tasks,
                  requires_human_audit=True, scope=str(Path(args.records)))
    write_json(out / "evaluation.json", report)
    write_jsonl(out / "test_predictions.jsonl", [{"id": r["id"], "task_id": r["task_id"],
                "probabilities": p, "class_id": r["supervision"]["class_id"]} for r, p in zip(test, test_probs)])
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", default="artifacts/text_probe_v2/frozen/records.jsonl")
    p.add_argument("--annotations", default=None, help="Optional separate draft labels for legacy trajectories")
    p.add_argument("--output", default="artifacts/text_probe_v2/model")
    p.add_argument("--base", default="models/Qwen3-0.6B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--pointer-size", type=int, default=256)
    p.add_argument("--readout", choices=["linear", "pointer"], default="linear")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-train-tokens", type=int, default=4096)
    p.add_argument("--max-eval-tasks", type=int, default=96, help="Fixed seeded task subset; 0 evaluates all tasks")
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())


if __name__ == "__main__":
    main()
