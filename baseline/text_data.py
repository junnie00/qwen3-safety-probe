"""Causal text observations: paired past drafts/targets, never the current target."""

from copy import deepcopy
import json


CLASS_NAMES = ("target_hijacked", "draft_hijacked_target_resisted", "both_normal")
EMAIL_MARKER = "Here is the content of the email:\n"


def context_from_case(case):
    if "context" in case:
        context = deepcopy(case["context"])
        if case.get("tools"):
            context.setdefault("available_tools", deepcopy(case["tools"]))
        return context
    trusted, users, environment, conversation = [], [], [], []
    for message in case["messages"]:
        role, content = message["role"], message.get("content", "")
        if role == "system" and EMAIL_MARKER in content:
            # The BIPIA builder puts an externally sourced email inside a system
            # message. Its dataset provenance, not the enclosing role, sets trust.
            instruction, email = content.split(EMAIL_MARKER, 1)
            trusted.append(instruction.strip())
            environment.append({"source": "email", "content": email})
        elif role in ("system", "developer"):
            trusted.append(content)
        elif role == "user":
            users.append(content)
        elif role == "tool":
            environment.append({"source": message.get("name", "tool"), "content": content})
        else:
            conversation.append(deepcopy(message))
    context = {"trusted_instructions": trusted, "user_requests": users,
               "untrusted_environment": environment, "prior_conversation": conversation}
    if len(case["messages"]) > 2:
        context["ordered_messages"] = deepcopy(case["messages"])
    if case.get("tools"):
        context["available_tools"] = deepcopy(case["tools"])
    return context


def source_messages(context):
    """A source-separated prompt for NEW collection, identical in both arms."""
    if "ordered_messages" in context:
        messages = []
        for message in deepcopy(context["ordered_messages"]):
            if message["role"] == "system" and EMAIL_MARKER in message.get("content", ""):
                trusted, email = message["content"].split(EMAIL_MARKER, 1)
                messages.extend([{"role": "system", "content": trusted.strip()},
                                 {"role": "tool", "name": "email", "content": email}])
            else:
                messages.append(message)
        return messages
    messages = [{"role": "system", "content": "\n".join(context["trusted_instructions"])}]
    messages.extend({"role": "user", "content": text} for text in context["user_requests"])
    messages.extend(deepcopy(context.get("prior_conversation", [])))
    messages.extend({"role": "tool", "name": item["source"], "content": item["content"]}
                    for item in context["untrusted_environment"])
    return messages


def observation(context, history_pairs, current_draft):
    pairs = []
    for pair in history_pairs:
        pairs.append({key: deepcopy(pair[key]) for key in
                      ("draft", "target", "accepted_draft_tokens", "proposed_draft_tokens")})
    return {"context": deepcopy(context), "history_pairs": pairs, "current_draft": current_draft}


def input_text(value):
    if set(value) != {"context", "history_pairs", "current_draft"}:
        raise ValueError("Probe input permits only context, past pairs, and current_draft")
    for pair in value["history_pairs"]:
        if set(pair) != {"draft", "target", "accepted_draft_tokens", "proposed_draft_tokens"}:
            raise ValueError("Past pair contains undeclared fields")
    # The growing history must precede the changing draft so its exact token
    # prefix can be reused by the causal backbone's KV cache.
    ordered = {key: value[key] for key in ("context", "history_pairs", "current_draft")}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def class_id(draft_hijacked, target_hijacked):
    if target_hijacked == 1:
        return 0  # Includes a normal draft followed by a hijacked target correction.
    if target_hijacked == 0 and draft_hijacked is not None:
        return 1 if draft_hijacked else 2
    return None


def trace_records(case, blocks, judgment=None):
    context, history, prefix, records = context_from_case(case), deepcopy(case.get("history_pairs", [])), "", []
    rule = case.get("evaluator", {})
    for index, block in enumerate(blocks):
        draft, target = block["draft_text"], block["target_new_text"]
        after, d, t = False, None, None
        if rule.get("kind") == "clean":
            d = t = 0
        elif rule.get("kind") == "marker":
            marker = rule["text"]
            after = marker in prefix
            d = int(marker in prefix + draft) if not after else None
            t = int(marker in prefix + target) if not after else None
        elif judgment is not None:
            truth, first = judgment["task_label"], judgment["first_affected_block"]
            if truth == 0:
                t = 0
            elif truth == 1 and first is not None:
                after = index > first
                t = int(index >= first)
        supervision = {"target_output": target, "target_prefix_before": prefix,
                       "target_text_so_far": prefix + target, "draft_hijacked": d,
                       "target_hijacked": t, "class_id": class_id(d, t),
                       "after_violation": after,
                       "task_label": None if judgment is None else judgment["task_label"]}
        records.append({"id": f"{case['id']}/block-{index}", "case_id": case["id"],
                        "task_id": case.get("task_id", case["id"]),
                        "group_id": case.get("group_id", case["id"]), "split": case["split"],
                        "input": observation(context, history, draft), "supervision": supervision})
        history.append({"draft": draft, "target": target,
                        "accepted_draft_tokens": block["accepted_draft_tokens"],
                        "proposed_draft_tokens": block["proposed_draft_tokens"]})
        prefix += target
    if judgment is None and rule.get("kind") in ("clean", "marker"):
        truth = int(rule.get("kind") == "marker" and rule["text"] in prefix)
        for record in records:
            record["supervision"]["task_label"] = truth
    return records
