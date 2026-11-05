"""对实际输入检查 batch/padding 数值稳定性；不读取标签做模型选择。"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from baseline.modernbert_data import json_lines, tokenize_view, view_fields
from baseline.modernbert_poc import encode_batch


def compare(a, b):
    return {'max_abs': float(np.abs(a - b).max()),
            'relative_l2': float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-10)),
            'cosine': float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))}


def main():
    if not os.environ.get('POC_CGROUP'):
        raise SystemExit('需要受限wrapper')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.5)
    info = json.loads((ROOT / 'artifacts/modernbert_oracle_poc/model.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(info['local_path'], local_files_only=True)
    cases = {c['id']: c for c in json_lines(ROOT / 'artifacts/text_probe_v2/frozen/resolved_cases.jsonl')}
    examples = {}
    for record in json_lines(ROOT / 'artifacts/text_probe_v2/frozen/records.jsonl'):
        fields = view_fields(record, cases[record['case_id']])
        for view in ['G', 'E']:
            if view == 'E' and not fields['attack']:
                continue
            ids, _ = tokenize_view(fields, view, tokenizer, 4096)
            if len(ids) <= 256 and ('short' not in examples or len(ids) > len(examples['short'])):
                examples['short'] = ids
            if 'long' not in examples or len(ids) > len(examples['long']):
                examples['long'] = ids
    config = AutoConfig.from_pretrained(info['local_path'], local_files_only=True)
    config.reference_compile = False
    model = AutoModel.from_pretrained(info['local_path'], config=config, local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    references = {}
    report = []
    for dtype in [torch.float32, torch.bfloat16]:
        model.to(dtype=dtype)
        for name, ids in examples.items():
            baseline = None
            for batch_size in [1, 2, 4, 8]:
                vector = encode_batch(model, tokenizer, [ids] * batch_size, 'cuda')[0]
                if baseline is None:
                    baseline = vector.copy()
                if dtype == torch.float32 and batch_size == 1:
                    references[name] = vector.copy()
                row = {'dtype': str(dtype), 'example': name, 'length': len(ids), 'batch': batch_size,
                       'versus_batch1': compare(vector, baseline), 'versus_fp32': compare(vector, references[name])}
                report.append(row)
                print(json.dumps(row), flush=True)
        short = examples['short']
        padded = encode_batch(model, tokenizer, [short, examples['long']], 'cuda')[0]
        row = {'dtype': str(dtype), 'example': 'short_padded_to_long',
               'versus_fp32': compare(padded, references['short'])}
        report.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / 'artifacts/modernbert_oracle_poc/numerics.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
