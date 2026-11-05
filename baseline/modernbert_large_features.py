"""large特征提取：严格沿用FP32容差，不一致的batch跳过而不终止全部测速。"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from baseline.modernbert_data import SERIALIZATION_VERSION, build_dataset, digest, file_sha256
from baseline.modernbert_poc import bucket, cache_features, encode_batch, memory_pressure, write_json


def benchmark_checked(model, tokenizer, tokens, out_dir):
    representatives = {}
    for view in ['G', 'E']:
        for ids in tokens[view]:
            if ids is None:
                continue
            b = bucket(len(ids))
            if b not in representatives:
                representatives[b] = [ids, ids]
            if len(ids) < len(representatives[b][0]):
                representatives[b][0] = ids
            if len(ids) > len(representatives[b][1]):
                representatives[b][1] = ids
    results, selected = [], {}
    for b, (short, long) in sorted(representatives.items()):
        reference_long = encode_batch(model, tokenizer, [long], 'cuda')[0]
        reference_short = encode_batch(model, tokenizer, [short], 'cuda')[0]
        candidates = []
        for size in [1, 2, 4, 8]:
            if memory_pressure()[0] or torch.cuda.mem_get_info()[0] < 3 * 1024**3:
                if size == 1:
                    raise RuntimeError('batch1资源余量不足')
                break
            try:
                torch.cuda.reset_peak_memory_stats()
                warm = encode_batch(model, tokenizer, [long] * size, 'cuda')[0]
                padded = encode_batch(model, tokenizer, [short] + [long] * (size - 1), 'cuda')[0]
                long_diff = float(np.abs(warm - reference_long).max())
                short_diff = float(np.abs(padded - reference_short).max())
                valid = np.allclose(warm, reference_long, atol=1e-4, rtol=1e-4) and np.allclose(padded, reference_short, atol=1e-4, rtol=1e-4)
                result = {'bucket': b, 'length': len(long), 'short_length': len(short), 'batch': size,
                          'max_batch_difference': long_diff, 'max_padding_difference': short_diff,
                          'relative_l2': float(np.linalg.norm(warm - reference_long) / max(np.linalg.norm(reference_long), 1e-12)),
                          'status': 'accepted' if valid else 'rejected_numerical_consistency'}
                if valid:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(3):
                        encode_batch(model, tokenizer, [long] * size, 'cuda')
                    torch.cuda.synchronize()
                    rate = 3 * size / (time.perf_counter() - start)
                    candidates.append((rate, size))
                    result.update(examples_per_second=rate, tokens_per_second=rate * len(long))
                result.update(cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                              cuda_peak_reserved=torch.cuda.max_memory_reserved(),
                              cgroup_memory=memory_pressure()[1])
                results.append(result)
                print('large受检测速 ' + json.dumps(result), flush=True)
            except torch.cuda.OutOfMemoryError:
                results.append({'bucket': b, 'batch': size, 'status': 'cuda_oom_backoff'})
                gc.collect()
                torch.cuda.empty_cache()
                break
        if not candidates:
            raise RuntimeError('没有通过正确性检查的batch')
        best = max(rate for rate, size in candidates)
        selected[str(b)] = min(size for rate, size in candidates if rate >= .95 * best)
    write_json(Path(out_dir) / 'batch_benchmark.json', {
        'selected_batches': selected, 'measurements': results,
        'tolerance': {'atol': 1e-4, 'rtol': 1e-4},
        'method': '长度桶最长输入吞吐＋最短输入padding一致性；只从通过检查的batch中选最优5%内最小batch'})
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-manifest', default='artifacts/modernbert_oracle_poc/model-large.json')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP'):
        raise SystemExit('必须使用受限wrapper')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('需要CUDA')
    torch.cuda.set_per_process_memory_fraction(.5)
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model_info = json.loads(Path(args.model_manifest).read_text())
    config = {'model': model_info, 'serialization': SERIALIZATION_VERSION, 'max_length': 4096,
              'pooling': 'last_hidden_nonpadding_mean_fp32',
              'backbone': 'frozen_eval_fp32_sdpa_reference_compile_false',
              'precision_reason': '与已核验base保持FP32；严格allclose容差，不一致的batch拒绝并回退',
              'torch': torch.__version__, 'code_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in
                                                       ['modernbert_large_features.py', 'modernbert_poc.py', 'modernbert_data.py']}}
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != config:
        raise ValueError('配置变更，不能混用缓存')
    write_json(out / 'config.json', config)
    tokenizer = AutoTokenizer.from_pretrained(model_info['local_path'], local_files_only=True)
    rows, tokens, audit = build_dataset('artifacts/text_probe_v2/frozen/records.jsonl',
        'artifacts/text_probe_v2/frozen/resolved_cases.jsonl', tokenizer, out, 4096)
    cfg = AutoConfig.from_pretrained(model_info['local_path'], local_files_only=True)
    cfg.reference_compile = False
    model = AutoModel.from_pretrained(model_info['local_path'], config=cfg, local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    parameters = sum(p.numel() for p in model.parameters())
    batches = (json.loads((out / 'batch_benchmark.json').read_text())['selected_batches']
               if (out / 'batch_benchmark.json').exists() else benchmark_checked(model, tokenizer, tokens, out))
    contract = {'config': config, 'data': audit['source_sha256'], 'manifest_hash': digest(rows),
                'tokenizer_revision': model_info['revision']}
    write_json(out / 'cache_contract.json', contract)
    for view in ['G', 'E']:
        cache_features(model, tokenizer, rows, tokens, view, out, contract, batches)
    write_json(out / 'completed.json', {'elapsed_seconds': time.perf_counter() - started,
        'config_hash': digest(config), 'parameters': parameters, 'hidden_size': cfg.hidden_size,
        'trainable_backbone_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    print(f'large特征已完成：{out}', flush=True)


if __name__ == '__main__':
    main()
