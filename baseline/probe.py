"""Reuse the upstream two-layer MLP, without importing its serving stack."""

import torch

from third_party.SafeSpec.safety_head.safety_head import SafetyHead


FEATURE_SOURCE = "draft_last_layer_predictor_mean"


def new_probe(hidden_size):
    return SafetyHead(hidden_size, dropout=0.1, mlp_hidden_ratio=0.5)


def load_probe(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["feature_source"] != FEATURE_SOURCE:
        raise ValueError("Checkpoint was not trained on this baseline's draft features")
    probe = new_probe(checkpoint["hidden_size"])
    probe.load_state_dict(checkpoint["state_dict"])
    return probe.eval(), checkpoint
