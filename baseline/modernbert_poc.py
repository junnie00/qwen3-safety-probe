"""受限、可恢复的 ModernBERT oracle 双视图 PoC。通过 scripts/run_probe_limited.py 启动。"""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from baseline.modernbert_data import build_dataset, digest, file_sha256, SERIALIZATION_VERSION
from baseline.modernbert_heads import run_heads


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def memory_pressure():
    p = Path(os.environ['POC_CGROUP'])
    current = int((p / 'memory.current').read_text())
    high = int((p / 'memory.high').read_text())
    return current >= high, current, high


def mean_pool(hidden, mask):
    # 用FP32累积，避免长序列BF16求和损失；只在GPU保留本batch临时张量。
    weights = mask.unsqueeze(-1).to(torch.float32)
    return (hidden.float() * weights).sum(1) / weights.sum(1).clamp_min(1)


def encode_batch(model, tokenizer, sequences, device):
    padded = tokenizer.pad({'input_ids': sequences}, padding=True, return_tensors='pt')
    batch = {key: value.to(device) for key, value in padded.items() if key in ['input_ids', 'attention_mask']}
    with torch.inference_mode():
        output = model(**batch).last_hidden_state
        pooled = mean_pool(output, batch['attention_mask'])
    return pooled.cpu().numpy()


def bucket(length):
    for size in [256, 512, 1024, 2048, 4096, 8192]:
        if length <= size:
            return size
    raise ValueError('超过模型长度支持')


def benchmark(model, tokenizer, tokens, out_dir):
    representatives = {}
    for view in ['G', 'E']:
        for ids in tokens[view]:
            if ids is None:
                continue
            b = bucket(len(ids))
            if b not in representatives or len(ids) > len(representatives[b]):
                representatives[b] = ids
    rows, chosen = [], {}
    # 模型之外仍有别的任务：限制本进程allocator上限，且预留显存。
    for b, ids in sorted(representatives.items()):
        baseline = None
        rates = []
        for batch_size in [1, 2, 4, 8]:
            pressure, current, high = memory_pressure()
            free, _ = torch.cuda.mem_get_info()
            if pressure or free < 3 * 1024**3:
                if batch_size == 1:
                    raise RuntimeError('batch=1之前资源余量不足，暂停')
                break
            try:
                torch.cuda.reset_peak_memory_stats()
                warm = encode_batch(model, tokenizer, [ids] * batch_size, 'cuda')
                if baseline is None:
                    baseline = warm[0].copy()
                max_diff = float(np.abs(warm[0] - baseline).max())
                if not np.allclose(warm[0], baseline, atol=1e-4, rtol=1e-4):
                    raise RuntimeError(f'batch输出不一致：maxdiff={max_diff}')
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(3):
                    encode_batch(model, tokenizer, [ids] * batch_size, 'cuda')
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                rate = batch_size * 3 / elapsed
                row = {'bucket': b, 'length': len(ids), 'batch': batch_size,
                       'examples_per_second': rate, 'tokens_per_second': rate * len(ids),
                       'max_batch_difference': max_diff,
                       'cuda_peak_allocated': torch.cuda.max_memory_allocated(),
                       'cuda_peak_reserved': torch.cuda.max_memory_reserved(),
                       'cgroup_memory': memory_pressure()[1]}
                rows.append(row)
                rates.append((rate, batch_size))
                print('测速 ' + json.dumps(row), flush=True)
            except torch.cuda.OutOfMemoryError:
                rows.append({'bucket': b, 'batch': batch_size, 'status': 'cuda_oom_backoff'})
                gc.collect()
                torch.cuda.empty_cache()
                break
        if not rates:
            raise RuntimeError(f'{b}长度桶无可用batch')
        # 选择吞吐距最好不超过5%的最小batch，避免为微小收益占更多显存。
        best = max(rate for rate, _ in rates)
        chosen[str(b)] = min(n for rate, n in rates if rate >= best * 0.95)
    result = {'selected_batches': chosen, 'measurements': rows,
              'method': '每个长度桶取最长输入；预热一次、计时三次；选择距最优吞吐5%内最小batch'}
    write_json(Path(out_dir) / 'batch_benchmark.json', result)
    return chosen


def cache_features(model, tokenizer, rows, tokens, view, out_dir, contract, batches):
    path = Path(out_dir)
    meta_path = path / f'{view}_cache.json'
    shape = (len(rows), model.config.hidden_size)
    contract_hash = digest(contract)
    expected = {'contract_hash': contract_hash, 'shape': list(shape), 'dtype': 'float32', 'view': view}
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != expected:
            raise RuntimeError('特征缓存合同不匹配，必须使用新run目录')
        features = np.lib.format.open_memmap(path / f'{view}_features.npy', mode='r+')
        done = np.lib.format.open_memmap(path / f'{view}_done.npy', mode='r+')
    else:
        features = np.lib.format.open_memmap(path / f'{view}_features.npy', mode='w+', dtype='float32', shape=shape)
        features[:] = np.nan
        features.flush()
        done = np.lib.format.open_memmap(path / f'{view}_done.npy', mode='w+', dtype='bool', shape=(len(rows),))
        done[:] = False
        done.flush()
        write_json(meta_path, expected)
    order = defaultdict(list)
    # 完全相同的输入可以复用一次编码，但标签/训练样本不去重、不修改。
    duplicates, by_hash = {}, {}
    for i, ids in enumerate(tokens[view]):
        if ids is None:
            continue
        h = rows[i][view + '_hash']
        if h in by_hash:
            duplicates[i] = by_hash[h]
        else:
            by_hash[h] = i
            if not done[i]:
                order[bucket(len(ids))].append(i)
    count = 0
    start = time.perf_counter()
    with (path / 'extraction_progress.jsonl').open('a') as log:
        for b in sorted(order):
            indices = sorted(order[b], key=lambda i: len(tokens[view][i]))
            batch_size, offset = int(batches[str(b)]), 0
            while offset < len(indices):
                pressure, current, high = memory_pressure()
                if pressure:
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    if memory_pressure()[0]:
                        raise RuntimeError(f'系统内存软阈值触发，已保存缓存，暂停：{current}/{high}')
                free, _ = torch.cuda.mem_get_info()
                if free < 3 * 1024**3:
                    torch.cuda.empty_cache()
                    if torch.cuda.mem_get_info()[0] < 3 * 1024**3:
                        raise RuntimeError('显存余量不足3GiB，已保存缓存，暂停')
                batch_indices = indices[offset:offset + batch_size]
                try:
                    vectors = encode_batch(model, tokenizer, [tokens[view][i] for i in batch_indices], 'cuda')
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    print(f'{view} bucket={b} OOM降batch={batch_size}', flush=True)
                    continue
                if not np.isfinite(vectors).all():
                    raise RuntimeError('特征含NaN/Inf')
                features[batch_indices] = vectors
                features.flush()
                done[batch_indices] = True
                done.flush()
                offset += len(batch_indices)
                count += len(batch_indices)
                if count // 200 != (count - len(batch_indices)) // 200 or offset == len(indices):
                    state = {'view': view, 'bucket': b, 'batch': batch_size,
                             'newly_encoded': count, 'elapsed_seconds': time.perf_counter() - start,
                             'cgroup_memory': memory_pressure()[1],
                             'cuda_allocated': torch.cuda.memory_allocated(),
                             'cuda_reserved': torch.cuda.memory_reserved()}
                    log.write(json.dumps(state) + '\n')
                    log.flush()
                    print('提取 ' + json.dumps(state), flush=True)
        for i, source in duplicates.items():
            if not done[source]:
                raise RuntimeError('重复输入缓存源缺失')
            features[i] = features[source]
            done[i] = True
        features.flush()
        done.flush()
    eligible = [i for i, ids in enumerate(tokens[view]) if ids is not None]
    if not done[eligible].all() or not np.isfinite(features[eligible]).all():
        raise RuntimeError('特征缓存不完整')
    info = {'view': view, 'eligible_rows': len(eligible), 'unique_inputs': len(by_hash),
            'duplicate_rows': len(duplicates), 'newly_encoded': count,
            'elapsed_seconds': time.perf_counter() - start}
    write_json(path / f'{view}_extraction.json', info)
    return features


def make_report(out, results, audit, batches, elapsed):
    lines = ['# ModernBERT 冻结表示双视图 PoC 结果', '',
             '这是离线块级 oracle 证据实验，不是实际攻击定位器或在线拦截结果。', '',
             f'- 有效块样本：{audit["eligible_records"]}；原始块记录：{audit["raw_records"]}。',
             f'- 各长度桶实际选用batch：`{batches}`。',
             f'- 本次pipeline总耗时：{elapsed:.1f} 秒（不含模型下载）。', '',
             '| seed | 配置 | A召回 | A误报率 | B召回 | macro-F1 |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for seed, entries in results['seeds'].items():
        for name in ['M0', 'M1', 'M2']:
            r = entries[name]['test']
            lines.append(f'| {seed} | {name} | {r["A_recall"]:.4f} | {r["A_fpr"]:.4f} | {r["B_recall"]:.4f} | {r["macro_f1"]:.4f} |')
    lines += ['', '## 解读边界', '',
              '- M0：通用头；M1：通用头＋攻击子集通用头；M2：通用头＋攻击证据头。',
              '- 5%误报预算只在validation校准；测试误报率可能超过5%，必须按实报告。',
              '- 主比较为M2相对M1；完整配对来源组bootstrap区间及新增检出/误报见 `heads/results.json`。',
              '- 输入冲突、裁剪、按领域/标签长度分布见 `data_audit.json` 与 `length_audit.jsonl`。',
              '- 同一来源组多个块相关，不能将块数当作独立攻击数量。',
              '- validation复用于早停/阈值选择，B样本少；三seed不替代更多独立来源数据。',
              '- 缓存后线性头速度不代表端到端在线检测速度。',
              '- CPU/内存限额与连续采样见启动wrapper的日志目录。',
              '- 当前target、history_pairs、标签与来源ID均未输入编码器。', '']
    (out / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--model-manifest', default='artifacts/modernbert_oracle_poc/model.json')
    parser.add_argument('--records', default='artifacts/text_probe_v2/frozen/records.jsonl')
    parser.add_argument('--cases', default='artifacts/text_probe_v2/frozen/resolved_cases.jsonl')
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--stage', choices=['audit', 'features', 'all'], default='all')
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP'):
        raise SystemExit('必须通过 run_probe_limited.py 验证硬限后启动')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model_info = json.loads(Path(args.model_manifest).read_text())
    head_config = {'lr': 1e-3, 'weight_decay': 1e-4, 'head_batch': 128,
                   'max_epochs': 100, 'patience': 10, 'seeds': [42, 43, 44],
                   'max_a_fpr': 0.05, 'bootstrap_iterations': 500}
    config = {'model': model_info, 'serialization': SERIALIZATION_VERSION,
              'max_length': args.max_length, 'pooling': 'last_hidden_nonpadding_mean_fp32',
              'backbone': 'frozen_eval_fp32_sdpa_reference_compile_false',
              'precision_reason': 'BF16跨batch短输入相对L2漂移约1%-2%；FP32诊断稳定，正式缓存用FP32',
              'head': head_config, 'torch': torch.__version__,
              'source_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in
                                ['modernbert_poc.py', 'modernbert_data.py', 'modernbert_heads.py']}}
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('run配置已改变，请使用新目录，不覆盖旧实验')
    write_json(config_path, config)
    tokenizer = AutoTokenizer.from_pretrained(model_info['local_path'], local_files_only=True)
    rows, tokens, audit = build_dataset(args.records, args.cases, tokenizer, out, args.max_length)
    print('数据审计 ' + json.dumps({k: audit[k] for k in ['raw_records', 'eligible_records', 'counts', 'attack_train']}), flush=True)
    if args.stage == 'audit':
        return
    if not torch.cuda.is_available():
        raise RuntimeError('本实验要求CUDA GPU')
    torch.cuda.set_per_process_memory_fraction(0.50)
    model_config = AutoConfig.from_pretrained(model_info['local_path'], local_files_only=True)
    model_config.reference_compile = False
    model = AutoModel.from_pretrained(model_info['local_path'], config=model_config,
        local_files_only=True, dtype=torch.float32, attn_implementation='sdpa').to('cuda')
    model.eval().requires_grad_(False)
    benchmark_path = out / 'batch_benchmark.json'
    if benchmark_path.exists():
        batches = json.loads(benchmark_path.read_text())['selected_batches']
    else:
        batches = benchmark(model, tokenizer, tokens, out)
    contract = {'config': config, 'data': audit['source_sha256'],
                'manifest_hash': digest(rows), 'tokenizer_revision': model_info['revision']}
    write_json(out / 'cache_contract.json', contract)
    features_g = cache_features(model, tokenizer, rows, tokens, 'G', out, contract, batches)
    features_e = cache_features(model, tokenizer, rows, tokens, 'E', out, contract, batches)
    del model, tokens
    gc.collect()
    torch.cuda.empty_cache()
    if args.stage == 'features':
        return
    results = run_heads(features_g, features_e, rows, out / 'heads', head_config)
    elapsed = time.perf_counter() - started
    make_report(out, results, audit, batches, elapsed)
    write_json(out / 'completed.json', {'elapsed_seconds': elapsed, 'time': time.time(),
                                       'config_hash': digest(config)})
    print(f'实验完成，报告：{out / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
