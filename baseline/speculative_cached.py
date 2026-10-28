"""KV-cached speculative sampling with causal paired-history text-probe inputs.

Sampling uses p/q acceptance, positive-residual correction, and a target bonus.
Both arms use the same KV-cache path. No estimated safety time is subtracted.
"""

from concurrent.futures import ThreadPoolExecutor
import codecs
import json
from time import perf_counter

import torch

from baseline.text_data import observation


class TextStream:
    """Decode ByteLevel tokens across blocks, including split UTF-8 characters."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        backend = getattr(tokenizer, "backend_tokenizer", None)
        backend_decoder = getattr(backend, "decoder", None)
        self.decoder = None
        if backend_decoder is not None and json.loads(backend_decoder.__getstate__())["type"] == "ByteLevel":
            from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
            self.byte_ids = {char: byte for byte, char in bytes_to_unicode().items()}
            self.special_ids = set(tokenizer.all_special_ids)
            self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def decode(self, ids, *, commit=False, final=False):
        if self.decoder is None:
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        decoder = self.decoder if commit else codecs.getincrementaldecoder("utf-8")("replace")
        if not commit:
            decoder.setstate(self.decoder.getstate())
        raw = bytearray()
        for token in ids:
            if token not in self.special_ids:
                raw.extend(self.byte_ids[c] for c in self.tokenizer.convert_ids_to_tokens(token))
        return decoder.decode(raw, final=final)


def distribution(logits, temperature, top_p):
    probabilities = (logits.float() / temperature).softmax(-1)
    if top_p < 1:
        sorted_p, order = probabilities.sort(descending=True)
        sorted_p = sorted_p.masked_fill(sorted_p.cumsum(-1) - sorted_p >= top_p, 0)
        probabilities = torch.zeros_like(probabilities).scatter(-1, order, sorted_p)
        probabilities /= probabilities.sum()
    return probabilities


def residual(p, q):
    value = (p - q.to(p.device)).clamp_min(0)
    mass = value.sum()
    if mass <= 0:
        raise ValueError("A rejected proposal needs positive target-minus-draft residual mass")
    return value / mass


def forward(model, ids, cache):
    device = next(model.parameters()).device
    past = 0 if cache is None else cache.get_seq_length()
    return model(input_ids=torch.tensor([ids], device=device), past_key_values=cache,
                 attention_mask=torch.ones(1, past + len(ids), device=device, dtype=torch.long),
                 use_cache=True, return_dict=True)


def sync(model):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate(draft, target, tokenizer, prefix, context, *, block_size=4,
             max_new_tokens=128, temperature=0.0, top_p=1.0, seed=42,
             eos_token_ids=(), probe=None, threshold=0.5, intercept=False, history_pairs=(),
             draft_temperature=None, verifier_fusion=False):
    draft_temperature = temperature if draft_temperature is None else draft_temperature
    if not prefix or block_size < 1 or max_new_tokens < 1 or temperature < 0 or draft_temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid speculative generation settings")
    if temperature and not draft_temperature:
        raise ValueError("A sampled target needs a nonzero draft distribution for rejection sampling")
    draft.requires_grad_(False).eval()
    target.requires_grad_(False).eval()
    if probe is not None:
        probe.reset()
    d_device, t_device = next(draft.parameters()).device, next(target.parameters()).device
    d_rng = torch.Generator(device=d_device).manual_seed(seed)
    t_rng = torch.Generator(device=t_device).manual_seed(seed + 1)
    a_rng = torch.Generator().manual_seed(seed + 2)
    output, blocks, history = [], [], list(history_pairs)
    text_stream = TextStream(tokenizer)
    d_cache = t_cache = None
    eos = set(eos_token_ids)
    executor = ThreadPoolExecutor(max_workers=1) if probe is not None else None
    sync(draft)
    sync(target)
    started = perf_counter()
    try:
        # At block boundaries caches contain the context except its pending last
        # token. Rejected suffixes are cropped out of BOTH model caches.
        if len(prefix) > 1:
            d_cache = forward(draft, prefix[:-1], None).past_key_values
            t_cache = forward(target, prefix[:-1], None).past_key_values
        while len(output) < max_new_tokens:
            token_context = list(prefix) + output
            remaining = max_new_tokens - len(output)
            proposed, qs = [], []
            pending = token_context[-1]
            for _ in range(min(block_size, remaining)):
                result = forward(draft, [pending], d_cache)
                d_cache = result.past_key_values
                logits = result.logits[0, -1]
                if draft_temperature:
                    q = distribution(logits, draft_temperature, top_p)
                    token = int(torch.multinomial(q, 1, generator=d_rng))
                    if temperature:
                        qs.append(q)
                else:
                    token = int(logits.argmax())
                proposed.append(token)
                pending = token
                if token in eos:
                    break
            draft_text = text_stream.decode(proposed, final=proposed[-1] in eos)
            value = observation(context, history, draft_text)
            submitted = perf_counter()
            future = executor.submit(probe, value) if executor else None

            sync(target)
            verify_start = perf_counter()
            result = forward(target, [token_context[-1]] + proposed, t_cache)
            t_cache = result.past_key_values
            logits = result.logits[0]
            accepted, verified = 0, []
            for i, token in enumerate(proposed):
                if temperature:
                    p, q = distribution(logits[i], temperature, top_p), qs[i].to(t_device)
                    agrees = float(torch.rand((), generator=a_rng)) < min(1.0, float(p[token] / q[token]))
                else:
                    agrees = token == int(logits[i].argmax())
                if not agrees:
                    replacement = (int(torch.multinomial(residual(p, q), 1, generator=t_rng))
                                   if temperature else int(logits[i].argmax()))
                    verified.append(replacement)
                    break
                verified.append(token)
                accepted += 1
                if token in eos:
                    break
            if accepted == len(proposed) and len(verified) < remaining and verified[-1] not in eos:
                bonus = (int(torch.multinomial(distribution(logits[-1], temperature, top_p), 1, generator=t_rng))
                         if temperature else int(logits[-1].argmax()))
                verified.append(bonus)
            verified = verified[:remaining]
            for i, token in enumerate(verified):
                if token in eos:
                    verified = verified[:i + 1]
                    break
            sync(target)
            verify_end = perf_counter()

            probe_result = future.result() if future is not None else None
            decided = perf_counter()
            risk = None
            if probe_result is not None:
                risk = probe_result["probabilities"][0]
                if verifier_fusion and accepted == len(proposed):
                    # A draft classified as hijacked cannot be made safe merely
                    # by predicting resistance when verification accepted it all.
                    risk = min(1.0, risk + probe_result["probabilities"][1])
            alarm = bool(risk is not None and risk >= threshold)
            blocked = bool(intercept and alarm)
            published = [] if blocked else verified
            target_text = text_stream.decode(verified, commit=True,
                                             final=verified[-1] in eos or len(verified) == remaining)
            record = {"index": len(blocks), "draft_text": draft_text, "target_new_text": target_text,
                      "draft_ids": proposed, "target_ids": verified, "published_ids": published,
                      "accepted_draft_tokens": min(accepted, len(verified)),
                      "proposed_draft_tokens": len(proposed), "input": value,
                      "probe": probe_result, "alarm": alarm, "intercepted": blocked,
                      "release_risk_score": risk, "verifier_fusion": verifier_fusion,
                      "verify_started_at": verify_start, "verify_finished_at": verify_end,
                      "probe_submit_at": submitted, "decision_at": decided,
                      "target_verify_ms": (verify_end - verify_start) * 1000,
                      "probe_wait_ms": (decided - verify_end) * 1000 if future else 0.0,
                      "probe_overlapped_target": bool(probe_result is not None and
                          probe_result["started_at"] < verify_end and probe_result["finished_at"] > verify_start),
                      "probe_finished_before_target": bool(probe_result is not None and
                          probe_result["finished_at"] <= verify_end)}
            blocks.append(record)
            output.extend(published)
            if blocked or (output and output[-1] in eos) or len(output) >= max_new_tokens:
                break
            history.append({"draft": draft_text, "target": target_text,
                            "accepted_draft_tokens": record["accepted_draft_tokens"],
                            "proposed_draft_tokens": len(proposed)})
            keep = len(prefix) + len(output) - 1
            t_cache.crop(keep)
            # After full acceptance the draft cache lacks the last proposed token;
            # the target bonus remains pending for the next block.
            if d_cache.get_seq_length() < keep:
                d_cache = forward(draft, [proposed[-1]], d_cache).past_key_values
            d_cache.crop(keep)
    finally:
        if executor:
            executor.shutdown(wait=True)
    sync(draft)
    sync(target)
    return {"output_ids": output, "output": tokenizer.decode(output, skip_special_tokens=True),
            "blocks": blocks, "elapsed_ms": (perf_counter() - started) * 1000,
            "intercepted": bool(blocks and blocks[-1]["intercepted"]),
            "draft_tokens": sum(b["proposed_draft_tokens"] for b in blocks),
            "accepted_tokens": sum(min(b["accepted_draft_tokens"], len(b["published_ids"])) for b in blocks),
            "target_agreed_tokens": sum(b["accepted_draft_tokens"] for b in blocks),
            "kv_cache": True, "sampling": "rejection_sampling" if temperature else "greedy"}
