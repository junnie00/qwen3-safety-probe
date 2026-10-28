"""Batch-one greedy speculative decoding reference, with an optional async CPU probe.

The target's argmax is authoritative. KV caching is deliberately disabled in both
the protected and unprotected reference paths. This is not an optimized serving
engine or a claim of zero overhead.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import torch


@dataclass
class Block:
    index: int
    feature: torch.Tensor | None
    proposed: list[int]
    accepted: int
    verified: list[int]
    published: list[int]
    score: float | None
    verify_ms: float
    probe_ms: float
    probe_wait_ms: float
    probe_overlapped: bool


@torch.inference_mode()
def _score(probe, feature):
    start = perf_counter()
    score = probe(feature.unsqueeze(0)).sigmoid().item()
    return score, start, perf_counter()


def _forward(model, token_ids, hidden=False):
    device = next(model.parameters()).device
    inputs = torch.tensor([token_ids], device=device, dtype=torch.long)
    return model(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        output_hidden_states=hidden,
        use_cache=False,
        return_dict=True,
    )


@torch.inference_mode()
def generate(draft, target, prefix, *, block_size=4, max_new_tokens=128,
             eos_token_ids=(), probe=None, threshold=0.5, intercept=False,
             capture_features=True):
    """Both models are frozen; the probe sees only draft predictor states.

    A predictor state is already available when the draft emits its next token.
    Mean pooling these states needs no additional draft or target forward pass.
    The probe finishes before any verified tokens in its block are published.
    """
    if not prefix or block_size < 1 or max_new_tokens < 1:
        raise ValueError("Require a nonempty prefix and positive token limits")
    draft.requires_grad_(False).eval()
    target.requires_grad_(False).eval()
    if probe is not None:
        probe.cpu().eval()
    eos = set(eos_token_ids)
    output, blocks = [], []
    stopped = False
    start = perf_counter()
    executor = ThreadPoolExecutor(max_workers=1) if probe is not None else None
    try:
        while len(output) < max_new_tokens:
            context = list(prefix) + output
            proposed, states = [], []
            remaining = max_new_tokens - len(output)
            for _ in range(min(block_size, remaining)):
                result = _forward(draft, context + proposed,
                                  hidden=capture_features or probe is not None)
                token = int(result.logits[0, -1].argmax())
                if capture_features or probe is not None:
                    states.append(result.hidden_states[-1][0, -1].detach())
                proposed.append(token)
                if token in eos:
                    break
            # Transfer/extraction time stays in the measured total wall time.
            feature = torch.stack(states).float().mean(0).cpu() if states else None
            future = executor.submit(_score, probe, feature) if executor else None

            verify_start = perf_counter()
            result = _forward(target, context + proposed, hidden=False)
            offset = len(context) - 1
            # The extra prediction supplies either a correction or a bonus token.
            choices = result.logits[0, offset:offset + len(proposed) + 1].argmax(-1).cpu().tolist()
            verify_end = perf_counter()
            accepted = 0
            for candidate, expected in zip(proposed, choices):
                if candidate != expected:
                    break
                accepted += 1
            verified = proposed[:accepted] + [choices[accepted]]
            verified = verified[:remaining]
            for i, token in enumerate(verified):
                if token in eos:
                    verified = verified[:i + 1]
                    break
            accepted = min(accepted, len(verified))

            score, probe_ms, overlap = None, 0.0, False
            wait_start = perf_counter()
            if future is not None:
                score, probe_start, probe_end = future.result()
                probe_ms = (probe_end - probe_start) * 1000
                overlap = probe_start < verify_end and probe_end > verify_start
            wait_ms = (perf_counter() - wait_start) * 1000 if future else 0.0
            stopped = bool(intercept and score is not None and score >= threshold)
            published = [] if stopped else verified
            blocks.append(Block(len(blocks), feature, proposed, accepted, verified,
                                published, score, (verify_end - verify_start) * 1000,
                                probe_ms, wait_ms, overlap))
            output.extend(published)
            if stopped or (output and output[-1] in eos):
                break
    finally:
        if executor:
            executor.shutdown(wait=True)
    return {
        "output_ids": output,
        "blocks": blocks,
        "intercepted": stopped,
        "elapsed_ms": (perf_counter() - start) * 1000,
        "draft_tokens": sum(len(b.proposed) for b in blocks),
        "accepted_tokens": sum(min(b.accepted, len(b.published)) for b in blocks),
        "target_agreed_tokens": sum(b.accepted for b in blocks),
    }
