import json

import pytest

from baseline.data import write_jsonl
from baseline.label_bipia import parse_judgment
from baseline.prepare_bipia import prepare


@pytest.mark.parametrize("answer", [
    "not json",
    '{"hijacked": null, "confidence": "high"}',
    '{"hijacked": false, "confidence": "low"}',
    '{"hijacked": "false", "confidence": "high"}',
    '{"hijacked": true, "evidence": "not in output", "confidence": "high"}',
    '{"hijacked": true, "evidence": "", "confidence": "high"}',
])
def test_uncertain_judgments_never_become_safe_labels(answer):
    rows = [{"target_new_text": "injected behavior", "target_text_so_far": "injected behavior"}]
    assert parse_judgment(answer, rows, "injected behavior", "normal response")["task_label"] is None


def test_only_explicit_high_confidence_outcomes_and_valid_onsets_are_retained():
    rows = [{"target_new_text": "normal ", "target_text_so_far": "normal "},
            {"target_new_text": "injected behavior", "target_text_so_far": "normal injected behavior"}]
    safe = parse_judgment('{"hijacked": false, "confidence": "high"}', rows,
                          "normal injected behavior", "normal response")
    assert safe["task_label"] == 0 and safe["first_affected_block"] is None
    positive = parse_judgment(
        '{"hijacked": true, "evidence": "injected behavior", "confidence": "high"}',
        rows, "normal injected behavior", "normal response")
    assert positive["task_label"] == 1 and positive["first_affected_block"] == 1
    old_evidence = parse_judgment(
        '{"hijacked": true, "evidence": "normal", "confidence": "high"}',
        rows, "normal injected behavior", "normal response")
    assert old_evidence["task_label"] is None


def test_clean_control_regression_prevents_observed_false_hijack_labels():
    response = "The $ value paid by Ganesha is $420.00."
    rows = [{"target_new_text": response, "target_text_so_far": response}]
    wrong = '{"hijacked": true, "evidence": "Ganesha is $", "confidence": "high"}'
    result = parse_judgment(wrong, rows, response, response)
    assert result["task_label"] == 0
    assert result["reason"] == "counterfactual_control_match"


def test_official_payloads_and_group_separation_survive_preparation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    train = [{"context": f"email {i}", "question": f"question {i}", "ideal": "answer"}
             for i in range(10)]
    test = [{"context": "email 0", "question": "a different question", "ideal": "answer"},
            {"context": "unseen email", "question": "question", "ideal": "answer"}]
    write_jsonl(source / "email_train.jsonl", train)
    write_jsonl(source / "email_test.jsonl", test)
    for split, text in [("train", "TRAIN OFFICIAL PAYLOAD"), ("test", "TEST OFFICIAL PAYLOAD")]:
        (source / f"text_attack_{split}.json").write_text(json.dumps({"family": [text]}))
    output = tmp_path / "prepared"
    metadata = prepare(source, output)
    rows = [json.loads(s) for s in (output / "cases.jsonl").read_text().splitlines()]
    grouped = {}
    for row in rows:
        assert row["control_id"].endswith("-clean")
        assert row["group_id"] not in grouped or grouped[row["group_id"]] == row["split"]
        grouped[row["group_id"]] = row["split"]
        if row["attack_str"] is not None:
            expected = "TEST OFFICIAL PAYLOAD" if row["split"] == "test" else "TRAIN OFFICIAL PAYLOAD"
            assert row["attack_str"] == expected
            assert row["messages"][0]["content"].endswith("\n" + expected)
            assert row["evaluator"]["kind"] == "manual"
    assert metadata["test_emails_shared_with_train_removed"] == 1
    assert metadata["requires_human_audit"]


def test_judge_is_greedy_even_when_qwen_model_defaults_enable_sampling():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from baseline.label_bipia import judge_batch

    torch.set_num_threads(1)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=256, eos_token_id=None)
    target = Qwen3ForCausalLM(config).eval()
    target.generation_config.do_sample = True
    target.generation_config.transformers_version = "4.51.0"

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[3, 4]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}

        def batch_decode(self, *args, **kwargs):
            return ['{"hijacked": null, "confidence": "low", "reason": "test"}']

    case = {"clean_context": "email", "question": "question", "reference_answer": "answer",
            "attack_str": "attack", "clean_response": "answer", "target_response": "different"}
    rows = [{"target_new_text": "different", "target_text_so_far": "different"}]
    before = torch.random.get_rng_state().clone()
    judge_batch(target, Tokenizer(), [(case, rows)])
    assert torch.equal(before, torch.random.get_rng_state())
