"""轻量复核正式产物：输入/代码哈希、缓存、检查点重放、混淆矩阵与资源记录。"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from baseline.modernbert_data import digest, file_sha256, json_lines
from baseline.modernbert_heads import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP'):
        raise SystemExit('请通过受限wrapper启动验证')
    torch.set_num_threads(2)
    run = Path(args.run_dir)
    assert not (run / 'INVALID.json').exists(), '无效调试运行'
    completed = json.loads((run / 'completed.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    assert digest(config) == completed['config_hash']
    for name, sha in config['source_sha256'].items():
        assert file_sha256(ROOT / 'baseline' / name) == sha, f'代码已变更：{name}'
    audit = json.loads((run / 'data_audit.json').read_text())
    for name, filename in [('records', 'records.jsonl'), ('cases', 'resolved_cases.jsonl')]:
        assert file_sha256(ROOT / 'artifacts/text_probe_v2/frozen' / filename) == audit['source_sha256'][name]
    rows = list(json_lines(run / 'manifest.jsonl'))
    counts = Counter(f'{r["split"]}/{r["y"]}' for r in rows)
    assert dict(counts) == audit['counts']
    contract = json.loads((run / 'cache_contract.json').read_text())
    assert digest(rows) == contract['manifest_hash']
    features = {}
    for view in ['G', 'E']:
        metadata = json.loads((run / f'{view}_cache.json').read_text())
        assert metadata['contract_hash'] == digest(contract)
        x = np.load(run / f'{view}_features.npy', mmap_mode='r')
        done = np.load(run / f'{view}_done.npy', mmap_mode='r')
        eligible = np.array([view == 'G' or r['attack'] for r in rows])
        assert list(x.shape) == metadata['shape']
        assert done[eligible].all() and np.isfinite(x[eligible]).all()
        features[view] = x
    result = json.loads((run / 'heads/results.json').read_text())
    predictions = list(json_lines(run / 'heads/test_predictions.jsonl'))
    test = np.array([i for i, row in enumerate(rows) if row['split'] == 'test'])
    yt = np.array([rows[i]['y'] for i in test])
    enabled = np.array([rows[i]['attack'] for i in test])
    verified = []
    for seed in config['head']['seeds']:
        probabilities = {}
        for name in ['G', 'G_aux', 'E']:
            ckpt = torch.load(run / 'heads' / f'{name}-seed-{seed}.pt', map_location='cpu', weights_only=True)
            assert ckpt['seed'] == seed and ckpt['name'] == name and ckpt['classes'] == ['A', 'B', 'C']
            head = torch.nn.Linear(ckpt['hidden_size'], 3)
            head.load_state_dict(ckpt['state_dict'])
            head.eval()
            x = features['E' if name == 'E' else 'G']
            with torch.inference_mode():
                probabilities[name] = head(torch.from_numpy(np.asarray(x[test]).copy())).softmax(-1).numpy()
        for name in ['M0', 'M1', 'M2']:
            report = result['seeds'][str(seed)][name]
            assert report['validation']['A_fpr'] <= .05 + 1e-12
            scores = probabilities['G'].copy()
            if name != 'M0':
                aux = probabilities['G_aux' if name == 'M1' else 'E']
                scores[enabled] = np.maximum(scores[enabled], aux[enabled])
            pred = predict(scores, report['tau_A'], report['tau_B'])
            logs = [r for r in predictions if r['seed'] == seed and r['configuration'] == name]
            assert len(logs) == len(test)
            assert [r['id'] for r in logs] == [rows[i]['id'] for i in test]
            assert np.array_equal(pred, [r['prediction'] for r in logs])
            assert np.allclose(scores[:, 0], [r['score_A'] for r in logs], atol=1e-6)
            assert np.allclose(scores[:, 1], [r['score_B'] for r in logs], atol=1e-6)
            cm = np.zeros((3, 3), dtype=int)
            for truth, prediction in zip(yt, pred):
                cm[truth, prediction] += 1
            assert cm.tolist() == report['test']['confusion']
            verified.append(f'{seed}/{name}')
    resources = list(json_lines(run / 'run-resources/resources.jsonl'))
    proof = json.loads((run / 'run-resources/resource_limits.json').read_text())
    exit_info = json.loads((run / 'run-resources/exit.json').read_text())
    assert exit_info['exit_code'] == 0
    peak = max(r['memory.peak'] for r in resources)
    assert peak <= proof['memory.max']
    assert proof['memory.swap.max'] == 0 and proof['cpu.max'] == '200000 100000'
    last_events = dict(line.split() for line in resources[-1]['memory.events'].splitlines())
    assert int(last_events['oom']) == 0 and int(last_events['oom_kill']) == 0
    cpu_first = dict(line.split() for line in resources[0]['cpu.stat'].splitlines())
    cpu_last = dict(line.split() for line in resources[-1]['cpu.stat'].splitlines())
    cpu_seconds = (int(cpu_last['usage_usec']) - int(cpu_first['usage_usec'])) / 1e6
    wall = resources[-1]['time'] - resources[0]['time']
    bench = json.loads((run / 'batch_benchmark.json').read_text())
    # allocator统计，不含驱动/CUDA context；不是整卡总占用。
    allocated = max(r.get('cuda_peak_allocated', 0) for r in bench['measurements'])
    reserved = max(r.get('cuda_peak_reserved', 0) for r in bench['measurements'])
    verified_info = {'verified_configurations': verified, 'test_rows_each': len(test),
                     'resource_peak_bytes': peak, 'memory_events': last_events,
                     'cpu_seconds': cpu_seconds, 'wall_seconds': wall,
                     'average_logical_cores': cpu_seconds / wall,
                     'benchmark_cuda_peak_allocated': allocated,
                     'benchmark_cuda_peak_reserved': reserved,
                     'pipeline_seconds': completed['elapsed_seconds'],
                     'selected_batches': bench['selected_batches'],
                     'checkpoint_files': sorted(p.name for p in (run / 'heads').glob('*.pt'))}
    (run / 'verification.json').write_text(json.dumps(verified_info, ensure_ascii=False, indent=2))
    print(json.dumps(verified_info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
