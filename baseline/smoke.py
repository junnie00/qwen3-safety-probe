"""CPU-only synthetic feature smoke test. Its scores are NOT safety results."""

import argparse
from pathlib import Path

import torch

from baseline.data import load_cache
from baseline.evaluate import evaluate
from baseline.probe import FEATURE_SOURCE
from baseline.train import train


def make_fixture():
    generator = torch.Generator().manual_seed(7)
    features, rows = [], []
    for split, groups in [("train", 12), ("val", 6), ("test", 6)]:
        for group in range(groups):
            for attacked in (0, 1):
                task = f"{split}-{group}-{attacked}"
                for block in range(3):
                    label = int(attacked and block == 2)
                    features.append(torch.randn(16, generator=generator) * 0.1 + (2 if label else -2))
                    rows.append({"id": f"{task}/{block}", "task_id": task,
                                 "group_id": f"{split}-{group}", "split": split,
                                 "label": label, "task_label": attacked, "after_violation": False})
    return {"features": torch.stack(features), "records": rows,
            "draft_model": "synthetic-not-Qwen3", "target_model": "synthetic-not-Qwen3",
            "feature_source": FEATURE_SOURCE, "collection_mode": "silent", "synthetic": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/smoke")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(make_fixture(), out / "features.pt")
    data = load_cache(out / "features.pt")
    train(data, out / "model", epochs=30, lr=0.01)
    evaluate(data, out / "model/probe.pt", out / "evaluation")
    print("SYNTHETIC INFRASTRUCTURE CHECK ONLY: no LLM or attack benchmark was evaluated.")


if __name__ == "__main__":
    main()
