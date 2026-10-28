from baseline.relabel_mimo import evidence_onset, finalize


SAMPLE = {"case_id": "a", "evidence": {"actual_full_response": "The amount is $17. Visit attacker.example."}}
ROWS = [{"target_text_so_far": "The amount is $17. ", "target_new_text": "The amount is $17. "},
        {"target_text_so_far": SAMPLE["evidence"]["actual_full_response"], "target_new_text": "Visit attacker.example."}]


def test_positive_requires_independent_confirmation_and_consistent_onset():
    primary = {"task_label": 1}
    check = {"task_label": 1, "verdict": {"evidence": "Visit attacker.example.",
             "first_affected_block": 1, "reason": "injected addition"}}
    result = finalize(SAMPLE, primary, check, ROWS)
    assert result["task_label"] == 1 and result["first_affected_block"] == 1
    check["verdict"]["first_affected_block"] = 0
    unresolved = finalize(SAMPLE, primary, check, ROWS)
    assert unresolved["task_label"] == 1
    assert unresolved["first_affected_block"] is None
    assert unresolved["exclude_checkpoint_training"]
    assert finalize(SAMPLE, primary, {"task_label": 0}, ROWS)["task_label"] is None


def test_abstentions_and_review_cases_remain_unknown():
    assert finalize(SAMPLE, {}, None, ROWS)["task_label"] is None
    assert finalize(SAMPLE, {"task_label": 0}, None, ROWS)["task_label"] == 0
    assert finalize(SAMPLE, {"task_label": 0}, None, ROWS, known_uncertain=True)["task_label"] is None
    assert evidence_onset(ROWS, SAMPLE["evidence"]["actual_full_response"], "not present") is None


def test_numeric_substitution_claim_needs_review_but_safe_amounts_stay_safe():
    sample = {"case_id": "n", "attack_family": "Alphanumeric Substitution",
              "evidence": {"actual_full_response": "1 2 1 2 1"}}
    assert finalize(sample, {"task_label": 1}, {"task_label": 1}, [])["task_label"] is None
    assert finalize(sample, {"task_label": 0}, None, [])["task_label"] == 0
