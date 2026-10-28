"""LiveCodeBench v6 prompts, extraction, and public + private test execution."""

import json
import os
from pathlib import Path
import subprocess
import sys

from baseline.official.backend import ProbeStopped
from baseline.official.native import ROOT, paths


def load():
    paths()
    # The official prompt module opens its few-shot files relative to cwd.
    previous = Path.cwd()
    try:
        os.chdir(ROOT / "third_party/LiveCodeBench")
        import lcb_runner.prompts.code_generation
    finally:
        os.chdir(previous)
    # Construct official dataclasses only for selected questions: decoding all
    # private tests needlessly consumes memory. No HF builder/network downloads.
    path = ROOT / "data/official/LiveCodeBench/v6/test6.jsonl"
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def problem(entry):
    paths()
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    return CodeGenerationProblem(**entry)


def prompt(entry):
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.prompts.code_generation import format_prompt_generation
    return format_prompt_generation(problem(entry), LMStyle.CodeQwenInstruct)


def run(backend, entry, output_path):
    paths()
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code
    from lcb_runner.prompts.code_generation import get_codeqwen_question_template_answer
    # Preserve the official raw Qwen prompt; expose the equivalent trusted roles
    # to the Probe, without leaking public/private expected outputs into it.
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": get_codeqwen_question_template_answer(problem(entry)).removesuffix(
                    "<|im_end|>\n<|im_start|>assistant\n")}]
    try:
        result = backend.complete(messages, formatted_prompt=prompt(entry) + "<think>\n\n</think>\n\n")
    except ProbeStopped:
        return {"intercepted": True, "code": "", "native_valid": False}
    code = extract_code(result["output"], LMStyle.CodeQwenInstruct)
    request = Path(output_path).with_suffix(".check_input.json")
    request.write_text(json.dumps({"sample": problem(entry).get_evaluation_sample(), "code": code}))
    env = {k: v for k, v in os.environ.items() if k.lower() not in
           {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run([sys.executable, "-m", "baseline.official.lcb", "--check", str(request)],
                               cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    check = json.loads(completed.stdout.splitlines()[-1])
    request.unlink()
    return {"intercepted": False, "code": code, **check,
            "finish_reason": result["finish_reason"],
            "public_tests": len(problem(entry).public_test_cases),
            "private_tests": len(problem(entry).private_test_cases)}


if __name__ == "__main__":
    paths()
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
    value = json.loads(Path(sys.argv[2]).read_text())
    tests, metadata = check_correctness(value["sample"], value["code"], timeout=6, debug=False)
    print(json.dumps({"native_valid": bool(tests) and all(x is True or x == 1 for x in tests),
                      "test_results": [int(x) for x in tests], "native_metadata": metadata}))
