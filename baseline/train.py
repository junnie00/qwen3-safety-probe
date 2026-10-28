"""Train checkpoint-level BCE; choose a task-level threshold on validation only."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from baseline.data import indices_for, load_cache, write_json
from baseline.metrics import calibrate, task_predictions
from baseline.probe import FEATURE_SOURCE, new_probe


def train(cache, output, *, epochs=20, batch_size=256, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    x = cache["features"].float()
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError("Expected a finite [checkpoints, draft_hidden_size] feature matrix")
    tr = indices_for(cache, "train", training=True)
    va = indices_for(cache, "val", require_checkpoint_labels=False)
    va_loss = indices_for(cache, "val", training=True)
    labels = torch.tensor([cache["records"][i]["label"] for i in tr], dtype=torch.float32)
    if set(labels.tolist()) != {0.0, 1.0}:
        raise ValueError("Training needs both target-safe and target-hijacked checkpoint labels")
    head = new_probe(x.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x[tr], labels), batch_size=batch_size, shuffle=True)
    val_labels = torch.tensor([cache["records"][i]["label"] for i in va_loss], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(epochs):
        head.train()
        total = 0.0
        for features, truth in loader:
            optimizer.zero_grad()
            loss = loss_fn(head(features).squeeze(-1), truth)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(truth)
        head.eval()
        with torch.inference_mode():
            val_loss = loss_fn(head(x[va_loss]).squeeze(-1), val_labels).item()
        history.append({"epoch": epoch + 1, "train_loss": total / len(tr), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    if best_state is None:
        raise ValueError("epochs must be positive")
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        probabilities = head(x[va]).squeeze(-1).sigmoid().tolist()
    tasks = task_predictions([cache["records"][i] for i in va], probabilities)
    calibration = calibrate([t["label"] for t in tasks.values()], [t["score"] for t in tasks.values()])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": head.state_dict(), "hidden_size": x.shape[1],
        "feature_source": FEATURE_SOURCE, "threshold": calibration["threshold"],
        "draft_model": cache["draft_model"], "target_model": cache["target_model"],
        "label_source": cache.get("label_source", "provided_checkpoint_annotations"),
        "requires_human_audit": bool(cache.get("requires_human_audit", False)),
        "train_group_ids": sorted({cache["records"][i]["group_id"] for i in tr}),
        "val_group_ids": sorted({cache["records"][i]["group_id"] for i in va}),
    }, out / "probe.pt")
    report = {"seed": seed, "epochs": epochs, "train_checkpoints": len(tr),
              "val_checkpoints": len(va), "best_val_bce": best_loss,
              "val_loss_checkpoints": len(va_loss),
              "val_loss_excludes_after_violation": True,
              "lr": lr, "batch_size": batch_size, "dropout": 0.1,
              "torch_version": str(torch.__version__),
              "feature_source": FEATURE_SOURCE, "validation": calibration,
              "history": history,
              "synthetic": bool(cache.get("synthetic", False))}
    report["label_source"] = cache.get("label_source", "provided_checkpoint_annotations")
    report["requires_human_audit"] = bool(cache.get("requires_human_audit", False))
    write_json(out / "training.json", report)
    print(f"Saved {out / 'probe.pt'}; validation feasible={calibration['feasible_on_validation']}")
    print(calibration)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--annotations", help="Edited annotations.jsonl; needed if labels were initially null")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    cache = load_cache(args.features, args.annotations)
    if cache.get("collection_mode", "silent") != "silent":
        parser.error("Training truth must come from an uninterrupted silent run")
    train(cache, args.output, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)


if __name__ == "__main__":
    main()
