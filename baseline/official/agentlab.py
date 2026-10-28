"""AgentLAB Task-Injection native environments, tools, and executable goals.

Attack bundles must declare their planner model; absence of a bundle
is a blocked attack evaluation, never a fabricated negative/security success.
"""

import json

from baseline.official.backend import ProbeStopped
from baseline.official.native import ROOT, agentlab_paths, bfcl_handler_class


def suite(name, version="v1.2.1"):
    agentlab_paths()
    from agentdojo.task_suite.load_suites import get_suite
    return get_suite(version, name)


def pipeline(backend):
    agentlab_paths()
    import yaml
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
    from agentdojo.agent_pipeline.errors import AbortAgentError
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, ToolsExecutionLoop
    from agentdojo.functions_runtime import FunctionCall
    from agentdojo.types import get_text_content_as_str, text_content_block_from_string

    extract = bfcl_handler_class()._extract_tool_calls

    class LocalLLM(BasePipelineElement):
        def query(self, query, runtime, env, messages, extra_args):
            ordered = []
            for msg in messages:
                item = {"role": msg["role"], "content": get_text_content_as_str(msg["content"] or [])}
                if msg["role"] == "assistant" and msg.get("tool_calls"):
                    item["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                        "name": c.function, "arguments": c.args}} for c in msg["tool_calls"]]
                if msg["role"] == "tool":
                    item.update(tool_call_id=msg["tool_call_id"], name=msg["tool_call"].function)
                    item["content"] = msg.get("error") or item["content"]
                ordered.append(item)
            tools = [{"type": "function", "function": {"name": f.name, "description": f.description,
                      "parameters": f.parameters.model_json_schema()}} for f in runtime.functions.values()]
            try:
                result = backend.complete(ordered, tools)
            except ProbeStopped as error:
                raise AbortAgentError(str(error), list(messages), env) from error
            text = result["output"].split("</think>")[-1].strip()
            calls = extract(text)
            functions = [FunctionCall(function=c["name"], args=c["arguments"],
                         id=f"call_{len(backend.calls)}_{i}") for i, c in enumerate(calls)]
            assistant = {"role": "assistant", "content": [text_content_block_from_string(text)],
                         "tool_calls": functions or None}
            return query, runtime, env, [*messages, assistant], extra_args

    class Pipeline(BasePipelineElement):
        name = "Qwen3-8B-speculative"

        def query(self, query, runtime, env, messages=None, extra_args=None):
            system = yaml.safe_load((ROOT / "third_party/AgentLAB/Task-Injection/agentdojo/src/agentdojo/data/system_messages.yaml").read_text())["default"]
            state = query, runtime, env, messages or [], extra_args or {}
            llm = LocalLLM()
            for element in [SystemMessage(system), InitQuery(), llm,
                            ToolsExecutionLoop([ToolsExecutor(), llm])]:
                state = element.query(*state)
            self.messages = state[3]
            return state

    return Pipeline()


def run(backend, entry, attack=None):
    native = suite(entry["suite"])
    agent = pipeline(backend)
    user = native.user_tasks[entry["user_task"]]
    injections = {} if attack is None else attack["injections"]
    injection_task = None if attack is None else native.injection_tasks[entry["injection_task"]]
    utility, security = native.run_task_with_pipeline(agent, user, injection_task, injections)
    # Upstream's misleadingly named `security` is TRUE on attacker goal success
    # when an injection task is supplied. Clean calls return True unconditionally.
    return {"native_valid": bool(utility), "native_utility": bool(utility),
            "target_hijacked": None if attack is None else bool(security),
            "security_scope": "clean_utility_only" if attack is None else "native_attacker_goal",
            "intercepted": any(c["intercepted"] for c in backend.calls),
            "messages": getattr(agent, "messages", []),
            "attack_model": None if attack is None else attack["attack_model"]}


def load_attacks(path):
    rows = [json.loads(line) for line in open(path)]
    for row in rows:
        if row.get("attack_model") not in {"gpt-5.1", "mimo-v2.6-flash"} or not row.get("injections"):
            raise ValueError("Expected declared GPT-5.1 or MiMo Task-Injection bundles with explicit injections")
    return {(r["suite"], r["user_task"], r["injection_task"]): r for r in rows}
