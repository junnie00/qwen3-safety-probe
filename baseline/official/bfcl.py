"""BFCL-v4 native Qwen formatter, multi-turn simulators, and official checkers."""

from copy import deepcopy
from types import SimpleNamespace
from time import perf_counter

from baseline.official.backend import ProbeStopped
from baseline.official.native import BFCL_MODEL, bfcl_handler_class, bfcl_registry

QwenFCHandler = bfcl_handler_class()


def load(category):
    bfcl_registry()
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry
    if category.startswith(("memory", "web_search")) or category == "format_sensitivity":
        raise ValueError("This adapter covers official single/multi-turn categories; web search, memory and format sensitivity need their own protocols")
    entries = load_dataset_entry(category)
    answers = {} if "irrelevance" in category or category == "live_relevance" else {
        r["id"]: r for r in load_ground_truth_entry(category)}
    return entries, answers


def handler(backend, run_id):
    from overrides import override
    from bfcl_eval.model_handler.base_handler import BaseHandler
    from bfcl_eval.constants.enums import ModelStyle

    class LocalHandler(QwenFCHandler):
        def __init__(self):
            # No HTTP client/server is needed for the in-process frozen models.
            BaseHandler.__init__(self, "Qwen/Qwen3-8B", backend.temperature, BFCL_MODEL, True)
            self.model_style = ModelStyle.OSSMODEL
            self.model_name_underline_replaced = run_id

        @override
        def _query_prompting(self, inference_data: dict):
            start = perf_counter()
            prompt = self._format_prompt(inference_data["message"], inference_data["function"])
            # Qwen3's native non-thinking generation suffix. Same in every arm.
            prompt += "<think>\n\n</think>\n\n"
            inference_data["inference_input_log"] = {"formatted_prompt": prompt}
            result = backend.complete(inference_data["message"], inference_data["function"],
                                      formatted_prompt=prompt)
            response = SimpleNamespace(choices=[SimpleNamespace(text=result["output"])],
                usage=SimpleNamespace(prompt_tokens=result["prompt_tokens"],
                                      completion_tokens=len(result["output_ids"])))
            return response, perf_counter() - start

    return LocalHandler()


def clear_simulators():
    # Upstream keeps task state in module globals; repetitions/arms must start fresh.
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
    for key in list(vars(multi_turn_utils)):
        if key.endswith("_instance"):
            delattr(multi_turn_utils, key)


def judge(entry, answer, response, model_handler):
    from bfcl_eval.constants.enums import Language
    category = entry["id"].rsplit("_", 1)[0]
    try:
        if category.startswith("multi_turn"):
            from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
                multi_turn_checker, multi_turn_irrelevance_checker)
            decoded = [[model_handler.decode_execute(step, False) for step in turn] for turn in response]
            clear_simulators()
            result = multi_turn_checker(decoded, answer["ground_truth"], entry, category, "native_qwen")
            if result["valid"] and ("miss_func" in category or "miss_param" in category):
                result = multi_turn_irrelevance_checker(decoded, answer["ground_truth"])
            return result
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        if "irrelevance" in category or category == "live_relevance":
            from bfcl_eval.eval_checker.eval_runner import _evaluate_single_relevance_entry
            return _evaluate_single_relevance_entry(model_handler, entry["id"], response,
                                                    entry, BFCL_MODEL, category)
        decoded = model_handler.decode_ast(response, Language.PYTHON, False)
        language = Language.JAVA if category == "simple_java" else Language.JAVASCRIPT if category == "simple_javascript" else Language.PYTHON
        return ast_checker(entry["function"], decoded, answer["ground_truth"], language, category, BFCL_MODEL)
    except (ValueError, KeyError, SyntaxError, IndexError, TypeError) as error:
        return {"valid": False, "error_type": "decode_or_check_error", "error": str(error)}


def run(backend, entry, answer, run_id):
    from bfcl_eval.utils import make_json_serializable
    clear_simulators()
    native = handler(backend, run_id)
    try:
        response, metadata = native.inference(deepcopy(entry), include_input_log=True, exclude_state_log=False)
    except ProbeStopped:
        return {"native_valid": False, "intercepted": True, "native_result": {"error_type": "probe_stop"}}
    result = judge(entry, answer, response, native)
    return make_json_serializable({"native_valid": bool(result["valid"]), "intercepted": False,
            "native_result": result, "response": response, "native_metadata": metadata})
