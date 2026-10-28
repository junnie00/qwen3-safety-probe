"""Import pinned sources without installing every unrelated model vendor SDK."""

import ast
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
BFCL_MODEL = "Qwen/Qwen3-8B-FC"


def paths():
    os.environ.setdefault("BFCL_PROJECT_ROOT", str(ROOT / "artifacts/official_benchmarks/bfcl_runtime"))
    for path in [ROOT / "third_party/BFCL", ROOT / "third_party/LiveCodeBench",
                 ROOT / "third_party/AgentLAB/Task-Injection/agentdojo/src",
                 ROOT / "artifacts/official_benchmarks/site-packages"]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def agentlab_paths():
    paths()
    # The upstream package __init__ eagerly imports all cloud vendors. Import
    # the actual native submodules lazily so local tools need no Cohere account.
    name = "agentdojo.agent_pipeline"
    if name not in sys.modules:
        module = ModuleType(name)
        module.__path__ = [str(ROOT / "third_party/AgentLAB/Task-Injection/agentdojo/src/agentdojo/agent_pipeline")]
        sys.modules[name] = module


def bfcl_handler_class():
    paths()
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
    return QwenFCHandler


def bfcl_registry():
    """Load the exact upstream Qwen entry, skipping imports of unused vendors.

    Checkers and inference stay untouched. Only the model registry is restricted
    to the single model being evaluated; its dot/underscore flag is unmodified.
    """
    handler = bfcl_handler_class()
    name = "bfcl_eval.constants.model_config"
    if name in sys.modules:
        return
    path = ROOT / "third_party/BFCL/bfcl_eval/constants/model_config.py"
    tree = ast.parse(path.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "ModelConfig")
    entry = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == BFCL_MODEL:
                    entry = value
    if entry is None:
        raise ValueError("Pinned BFCL source has no requested Qwen model entry")
    module = ModuleType(name)
    module.__dict__.update(dataclass=dataclass, Optional=Optional, QwenFCHandler=handler)
    sys.modules[name] = module
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), module.__dict__)
    config = eval(compile(ast.Expression(entry), str(path), "eval"), module.__dict__)
    module.MODEL_CONFIG_MAPPING = {BFCL_MODEL: config}
