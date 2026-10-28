import json

import torch

from baseline.data import write_jsonl
from baseline.export_text import export_once, text_case


def fixture():
    case = {"id": "case", "messages": [{"role": "user", "content": "original task"}],
            "attack_name": "attack", "attack_str": "untrusted instruction"}
    rows = [
        {"id": "case/block-0", "case_id": "case", "group_id": "group", "split": "train",
         "draft_text": "rejected malicious draft", "target_new_text": "safe ",
         "target_text_so_far": "safe ", "accepted_draft_tokens": 0, "proposed_draft_tokens": 4},
        {"id": "case/block-1", "case_id": "case", "group_id": "group", "split": "train",
         "draft_text": "next draft", "target_new_text": "answer",
         "target_text_so_far": "safe answer", "accepted_draft_tokens": 2, "proposed_draft_tokens": 4},
    ]
    run = {"id": "case", "output": "safe answer"}
    judgment = {"case_id": "case", "task_label": 0, "first_affected_block": None}
    return case, rows, run, judgment


def test_current_target_output_never_leaks_into_text_probe_input():
    case, rows, run, judgment = fixture()
    result = text_case(case, rows, run, judgment, "draft", "target")
    assert result["input"]["messages"] == case["messages"]
    first, second = result["draft_steps"]
    assert first["input"]["accepted_prefix_text"] == ""
    assert second["input"]["accepted_prefix_text"] == "safe "
    assert first["input"]["draft_text"] == "rejected malicious draft"
    assert first["supervision"]["checkpoint_label"] == 0
    assert first["supervision"]["accepted_draft_tokens"] == 0
    assert "target_new_text" not in first["input"]
    assert result["supervision"]["target_response"] == "safe answer"


def test_unknown_text_labels_are_not_converted_to_safe():
    case, rows, run, judgment = fixture()
    judgment["task_label"] = None
    result = text_case(case, rows, run, judgment, "draft", "target")
    assert result["supervision"]["task_label"] is None
    assert all(s["supervision"]["checkpoint_label"] is None for s in result["draft_steps"])


def test_export_waits_for_complete_judgments_and_emits_no_vectors(tmp_path):
    case, rows, run, judgment = fixture()
    folder = tmp_path / "run/shards/00000"
    folder.mkdir(parents=True)
    torch.save({"features": torch.ones(2, 16), "records": rows,
                "draft_model": "draft", "target_model": "target"}, folder / "features.pt")
    write_jsonl(folder / "runs.jsonl", [run])
    write_jsonl(folder / "judgments.jsonl", [])
    output = tmp_path / "text"
    before = export_once({"case": case}, tmp_path / "run", output)
    assert before["exported_cases"] == 0
    write_jsonl(folder / "judgments.jsonl", [judgment])
    after = export_once({"case": case}, tmp_path / "run", output)
    assert after["exported_cases"] == 1 and after["exported_checkpoints"] == 2
    saved = json.loads((output / "shards/00000.jsonl").read_text())
    assert "features" not in saved
    assert saved["draft_steps"][0]["input"]["draft_text"] == rows[0]["draft_text"]
