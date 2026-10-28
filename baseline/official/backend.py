"""One frozen speculative decoder, with paired history scoped to one episode."""

from copy import deepcopy
from time import perf_counter


class ProbeStopped(Exception):
    """Terminate the native episode before parsing/executing a blocked call."""


class Backend:
    def __init__(self, tokenizer, draft, target, probe, config, *, block_size=4,
                 max_new_tokens=2048, temperature=0.0, seed=42):
        self.tokenizer, self.draft, self.target = tokenizer, draft, target
        self.probe, self.config = probe, config
        self.block_size, self.max_new_tokens = block_size, max_new_tokens
        self.temperature, self.seed = temperature, seed
        self.reset("unprotected")

    def reset(self, mode, seed=None):
        if mode not in {"unprotected", "silent", "intercept"}:
            raise ValueError(mode)
        self.mode, self.history, self.calls = mode, [], []
        self.run_seed = self.seed if seed is None else seed

    def complete(self, messages, tools=None, *, formatted_prompt=None, max_tokens=None):
        from baseline.speculative_cached import generate

        started = perf_counter()
        # Native task prompts are preserved. Tool feedback keeps its own role;
        # it is never promoted to trusted user/system instructions for the Probe.
        context = {"ordered_messages": deepcopy(messages)}
        if tools:
            context["available_tools"] = deepcopy(tools)
        if formatted_prompt is None:
            prefix = self.tokenizer.apply_chat_template(
                messages, tools=tools or None, tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
        else:
            prefix = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        limit = min(self.draft.config.max_position_embeddings,
                    self.target.config.max_position_embeddings)
        budget = min(self.max_new_tokens, max_tokens or self.max_new_tokens)
        if len(prefix) + budget > limit:
            raise ValueError(f"Native prompt {len(prefix)} + output budget {budget} exceeds {limit}; no truncation")
        eos = self.target.config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos] if eos is not None else []
        result = generate(self.draft, self.target, self.tokenizer, prefix, context,
                          block_size=self.block_size, max_new_tokens=budget,
                          temperature=self.temperature, seed=self.run_seed + len(self.calls),
                          eos_token_ids=eos, probe=self.probe if self.mode != "unprotected" else None,
                          threshold=self.config["threshold"], intercept=self.mode == "intercept",
                          history_pairs=self.history,
                          verifier_fusion=self.config.get("verifier_fusion", False))
        result.update(prompt_tokens=len(prefix), request_ms=(perf_counter() - started) * 1000,
                      finish_reason="probe_stop" if result["intercepted"] else
                      "stop" if result["output_ids"] and result["output_ids"][-1] in eos else "length")
        self.calls.append(result)
        for block in result["blocks"]:
            if not block["intercepted"]:
                self.history.append({"draft": block["draft_text"], "target": block["target_new_text"],
                                     "accepted_draft_tokens": block["accepted_draft_tokens"],
                                     "proposed_draft_tokens": block["proposed_draft_tokens"]})
        if result["intercepted"]:
            # A partial tool call must never reach the benchmark's tool executor.
            raise ProbeStopped("Probe stopped this episode before tool execution")
        return result
