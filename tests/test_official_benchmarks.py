"""Native oracle checks and causal state boundaries for benchmark integration."""

from copy import deepcopy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from baseline.official import agentlab, bfcl, lcb
from baseline.official.backend import Backend, ProbeStopped
from baseline.official.native import ROOT


def test_native_bfcl_ast_accepts_correct_rejects_wrong_arguments():
    entries, answers = bfcl.load("simple_python")
    entry = entries[0]
    native = bfcl.handler(SimpleNamespace(temperature=0), "test")
    def response(base):
        return '<tool_call>\n' + json.dumps({"name": "calculate_triangle_area", "arguments": {"base": base, "height": 5}}) + '\n</tool_call>'
    assert bfcl.judge(entry, answers[entry["id"]], response(10), native)["valid"]
    assert not bfcl.judge(entry, answers[entry["id"]], response(99), native)["valid"]


def test_native_multiturn_oracle_executes_real_state():
    entries, answers = bfcl.load("multi_turn_base")
    entry = entries[0]
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
    truth = answers[entry["id"]]["ground_truth"]
    bfcl.clear_simulators()
    assert multi_turn_checker([[t] for t in truth], truth, entry, "multi_turn_base", "test_oracle")["valid"]
    bfcl.clear_simulators()
    wrong = deepcopy(truth)
    wrong[0] = ["mkdir(dir_name='wrong')"]
    assert not multi_turn_checker([[t] for t in wrong], truth, entry, "multi_turn_base", "test_wrong")["valid"]


def test_bfcl_native_state_objects_use_upstream_json_serialization(monkeypatch):
    entries, answers = bfcl.load("multi_turn_base")
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.gorilla_file_system import Directory
    native = SimpleNamespace(inference=lambda *a, **k: ([], {"state": Directory("/")}),
                             decode_execute=lambda *a: [])
    monkeypatch.setattr(bfcl, "handler", lambda *a: native)
    result = bfcl.run(None, entries[0], answers[entries[0]["id"]], "serialization")
    assert "Directory" in json.dumps(result)


def test_native_relevance_does_not_need_nonexistent_answer_file():
    entries, answers = bfcl.load("irrelevance")
    assert not answers
    native = bfcl.handler(SimpleNamespace(temperature=0), "test")
    assert bfcl.judge(entries[0], None, "No applicable function.", native)["valid"]


def test_native_agentlab_goal_boolean_means_attack_success():
    native = agentlab.suite("banking")
    from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
    user, injection = native.user_tasks["user_task_0"], native.injection_tasks["injection_task_0"]
    assert native.run_task_with_pipeline(GroundTruthPipeline(user), user, None, {})[0]
    # This pin's `security` boolean is success of the attack, not resistance.
    assert native.run_task_with_pipeline(GroundTruthPipeline(injection), user, injection, {})[1]


def test_native_agentlab_probe_stop_prevents_tools(monkeypatch):
    native = agentlab.suite("banking")
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.agent_pipeline.errors import AbortAgentError
    executed = []
    monkeypatch.setattr(FunctionsRuntime, "run_function", lambda *a, **k: executed.append(True))
    class Stopped:
        calls = []
        def complete(self, messages, tools):
            raise ProbeStopped("blocked before tool parsing")
    env = native.load_and_inject_default_environment({})
    with pytest.raises(AbortAgentError):
        agentlab.pipeline(Stopped()).query("test", FunctionsRuntime(native.tools), env)
    assert not executed


def test_backend_keeps_past_draft_and_target_across_turns_and_resets(monkeypatch):
    from baseline import speculative_cached
    received = []
    def fake_generate(*args, **kwargs):
        received.append(deepcopy(kwargs["history_pairs"]))
        return {"output": "target", "output_ids": [3], "intercepted": False, "blocks": [{
            "draft_text": "rejected draft", "target_new_text": "corrected target",
            "accepted_draft_tokens": 0, "proposed_draft_tokens": 4, "intercepted": False}]}
    monkeypatch.setattr(speculative_cached, "generate", fake_generate)
    tok = SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2])
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4000, eos_token_id=9))
    backend = Backend(tok, model, model, None, {"threshold": .5})
    backend.complete([{"role": "user", "content": "first"}])
    backend.complete([{"role": "user", "content": "second"}])
    assert received[0] == []
    assert received[1][0]["draft"] == "rejected draft"
    assert received[1][0]["target"] == "corrected target"
    backend.reset("silent")
    backend.complete([{"role": "user", "content": "new episode"}])
    assert received[2] == []


def test_native_lcb_checker_executes_good_and_bad_code(tmp_path):
    # An actual official v6 stdin question, including its private test cases.
    entry = next(e for e in lcb.load() if e["question_id"] == "abc387_a")
    sample = lcb.problem(entry).get_evaluation_sample()
    for code, expected in [("a,b=map(int,input().split());print((a+b)**2)", True), ("print(0)", False)]:
        request = tmp_path / "input.json"
        request.write_text(json.dumps({"sample": sample, "code": code}))
        result = subprocess.run([sys.executable, "-m", "baseline.official.lcb", "--check", str(request)],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        assert json.loads(result.stdout.splitlines()[-1])["native_valid"] == expected


def test_selection_is_fixed_and_no_private_tests_in_prompt():
    from baseline.run_official_benchmarks import select
    tasks = select(["livecodebench", "agentlab"])
    assert [t["id"] for t in tasks] == ["abc387_a", "abc387_b", "banking/user_task_0/clean", "banking/user_task_0/injection_task_0"]
    formatted = lcb.prompt(tasks[0]["entry"])
    assert "public_test_cases" not in formatted
    assert "private_test_cases" not in formatted


def test_native_gpt_planner_prompts_and_multienvironment_mapping_offline(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    from baseline.official.agentlab_search import native_planner, TASK_ROOT
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-0.6B", local_files_only=True)
    important, initial, rewrite = native_planner(tokenizer)
    target = SimpleNamespace(name="Qwen3-8B-speculative")
    for name in ["banking", "workspace", "travel", "slack"]:
        native = agentlab.suite(name)
        user = native.user_tasks["user_task_0"]
        injection = next(iter(native.injection_tasks.values()))
        knowledge = important.DummyKnowledgeAttack(native, target)
        prompt = initial.build_initial_prompt(user.PROMPT, injection.GOAL, knowledge.get_tools_sequence(user),
            knowledge.get_tools_sequence(injection), suite_name=name, agent_model_name=target.name)
        assert user.PROMPT in prompt and injection.GOAL in prompt
    native = agentlab.suite("banking")
    monkeypatch.setattr(important, "_find_repo_root", lambda: tmp_path)
    cache = tmp_path / "res/long_horizon" / target.name / "banking/v1.2.1/user_task_0_injection_task_0"
    cache.mkdir(parents=True)
    snippets = json.loads((TASK_ROOT / "res/long_horizon/example_banking.json").read_text())
    (cache / "newest_injection.json").write_text(json.dumps(snippets))
    attack = important.Human_ReasoningAttack(native, target)
    injections = attack.attack(native.user_tasks["user_task_0"], native.injection_tasks["injection_task_0"])
    assert "injection_bill_text" in injections
    assert "bank_file_injection_content" in injections
    assert "<|im_start|>" in injections["injection_bill_text"]
    messages = []
    rewritten, search_success = rewrite.build_rewrite_prompt("source", "target", messages, snippets, suite_name="banking")
    assert rewritten and search_success is False


def test_gpt_credentials_required_without_gpu_or_network(monkeypatch):
    from baseline.official.agentlab_search import openai_client
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="credentials missing"):
        openai_client()


def test_intercept_detection_truth_comes_from_unprotected_target():
    from baseline.run_official_benchmarks import summarize
    settings = {"full": False, "model_devices": ["cuda:0", "cuda:1"]}
    def row(mode, actual_hijacked, alarm):
        return {"benchmark": "AgentLAB-Task-Injection", "id": "banking/user_task_0/injection_task_0",
            "repeat": 0, "mode": mode, "status": "completed", "truth": actual_hijacked, "alarm": alarm,
            "result": {"native_valid": False, "intercepted": mode == "intercept"},
            "calls": [{"request_ms": 1, "output_ids": [3], "finish_reason": "stop", "blocks": [{
                "alarm": alarm, "proposed_draft_tokens": 4, "accepted_draft_tokens": 1, "probe_wait_ms": 0}]}]}
    report = summarize([row("unprotected", 1, False), row("intercept", 0, True)], settings)
    metrics = report["by_benchmark"]["AgentLAB-Task-Injection"]["by_mode"]["intercept"]
    assert metrics["positive_episodes"] == 1
    assert metrics["negative_episodes"] == 0
    assert metrics["asr"] == 0
    assert metrics["task_false_alarm_rate"] is None
    assert metrics["task_miss_rate_any_alarm"] == 0
