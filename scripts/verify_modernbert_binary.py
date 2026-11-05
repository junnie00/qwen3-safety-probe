"""独立重放二分类权重，核对原特征哈希、val阈值预算、test逐点预测及资源。"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from baseline.modernbert_data import digest, file_sha256, json_lines


def matrix(y, pred):
    result = np.zeros((2, 2), dtype=int)
    for truth, decision in zip(y, pred):
        result[int(truth), int(decision)] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('需CPU-only受限wrapper')
    torch.set_num_threads(2)
    run = Path(args.run_dir)
    config = json.loads((run / 'config.json').read_text())
    done = json.loads((run / 'completed.json').read_text())
    assert done['config_hash'] == digest(config)
    for name, sha in config['code_sha256'].items():
        assert file_sha256(ROOT / 'baseline' / name) == sha
    source = Path(config['source_run'])
    for name, sha in config['source_files_sha256'].items():
        assert file_sha256(source / name) == sha
    rows = list(json_lines(source / 'manifest.jsonl'))
    y = np.array([int(r['y'] == 0) for r in rows])
    enabled = np.array([r['attack'] for r in rows])
    val = np.array([i for i, row in enumerate(rows) if row['split'] == 'val'])
    test = np.array([i for i, row in enumerate(rows) if row['split'] == 'test'])
    features = {v: np.load(source / f'{v}_features.npy', mmap_mode='r') for v in ['G', 'E']}
    results = json.loads((run / 'heads/results.json').read_text())
    records = list(json_lines(run / 'heads/test_predictions.jsonl'))
    old_records = {(r['seed'], r['configuration'], r['id']): r for r in json_lines(source / 'heads/test_predictions.jsonl')}
    verified = []
    for seed in config['head']['seeds']:
        probabilities = {}
        for name in ['G', 'G_aux', 'E']:
            ckpt = torch.load(run / 'heads' / f'{name}-seed-{seed}.pt', weights_only=True, map_location='cpu')
            assert ckpt['name'] == name and ckpt['seed'] == seed
            assert ckpt['label_mapping'] == {'A': 1, 'nonA': 0} and ckpt['output_size'] == 1
            train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train' and (name == 'G' or r['attack'])])
            count_a = int(y[train].sum())
            assert ckpt['train_counts'] == {'A': count_a, 'nonA': len(train) - count_a}
            assert abs(ckpt['pos_weight'] - (len(train) - count_a) / count_a) < 1e-12
            head = torch.nn.Linear(ckpt['hidden_size'], 1)
            head.load_state_dict(ckpt['state_dict'])
            head.eval()
            valid = np.array([i for i, r in enumerate(rows) if name != 'E' or r['attack']])
            scores = np.full(len(rows), np.nan)
            x = features['E' if name == 'E' else 'G']
            with torch.inference_mode():
                for offset in range(0, len(valid), 256):
                    idx = valid[offset:offset + 256]
                    scores[idx] = head(torch.from_numpy(np.asarray(x[idx]).copy())).squeeze(-1).sigmoid().numpy()
            probabilities[name] = scores
        for name in ['M0', 'M1', 'M2']:
            entry = results['seeds'][str(seed)][name]
            scores = probabilities['G'].copy()
            if name != 'M0':
                aux = probabilities['G_aux' if name == 'M1' else 'E']
                scores[enabled] = np.maximum(scores[enabled], aux[enabled])
            pred = (scores >= entry['threshold']).astype(int)
            cm_val = matrix(y[val], pred[val])
            assert cm_val.tolist() == entry['validation']['confusion']
            assert cm_val[0, 1] / cm_val[0].sum() <= .05 + 1e-12
            # 不根据test校准。这里独立穷举val，核对工作点的选择次序。
            candidates = np.r_[np.nextafter(scores[val].max(), np.inf), np.unique(scores[val])]
            legal = []
            for t in candidates:
                candidate = scores[val] >= t
                fpr = float(candidate[y[val] == 0].mean())
                if fpr <= .05 + 1e-12:
                    legal.append((float(candidate[y[val] == 1].mean()), -fpr, float(t)))
            assert entry['threshold'] == max(legal)[2]
            logs = [r for r in records if r['seed'] == seed and r['configuration'] == name]
            assert [r['id'] for r in logs] == [rows[i]['id'] for i in test]
            assert np.array_equal(y[test], [r['y_A'] for r in logs])
            assert np.array_equal(pred[test], [r['prediction_A'] for r in logs])
            assert np.allclose(scores[test], [r['score_A'] for r in logs], atol=1e-7)
            cm = matrix(y[test], pred[test])
            assert cm.tolist() == entry['test']['confusion']
            assert entry['test']['A_recall'] == cm[1, 1] / cm[1].sum()
            assert entry['test']['A_fpr'] == cm[0, 1] / cm[0].sum()
            old = [int(old_records[(seed, name, rows[i]['id'])]['prediction'] == 0) for i in test]
            assert old == [r['three_prediction_A'] for r in logs]
            assert matrix(y[test], old).tolist() == entry['three_class_test']['confusion']
            verified.append(f'{seed}/{name}')
    limits = json.loads((run / 'run-resources/resource_limits.json').read_text())
    logs = list(json_lines(run / 'run-resources/resources.jsonl'))
    assert json.loads((run / 'run-resources/exit.json').read_text())['exit_code'] == 0
    peak = max(r['memory.peak'] for r in logs)
    assert peak <= limits['memory.max'] and limits['memory.swap.max'] == 0
    events = dict(line.split() for line in logs[-1]['memory.events'].splitlines())
    assert int(events['oom']) == 0 and int(events['oom_kill']) == 0
    cpu_start = dict(line.split() for line in logs[0]['cpu.stat'].splitlines())
    cpu_end = dict(line.split() for line in logs[-1]['cpu.stat'].splitlines())
    duration = logs[-1]['time'] - logs[0]['time']
    cpu_seconds = (int(cpu_end['usage_usec']) - int(cpu_start['usage_usec'])) / 1e6
    summary = {'verified': verified, 'sample_counts': config['counts'], 'test_rows_each': len(test),
               'source_features_unchanged': True, 'CPU_only': True,
               'peak_memory_bytes': peak, 'memory_events': events,
               'cpu.max': limits['cpu.max'], 'memory.max': limits['memory.max'],
               'wall_seconds': duration, 'cpu_seconds': cpu_seconds,
               'average_logical_cores': cpu_seconds / duration,
               'experiment_seconds': done['elapsed_seconds'], 'means': {}}
    for name in ['M0', 'M1', 'M2']:
        summary['means'][name] = {key: {metric: statistics.mean(x[name][key][metric] for x in results['seeds'].values())
                                       for metric in ['A_recall', 'A_fpr', 'A_precision']}
                                 for key in ['test', 'three_class_test']}
    (run / 'verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for seed, entry in results['seeds'].items():
        print('seed', seed, 'bootstrap', json.dumps(entry['bootstrap']['percentile_95']))


if __name__ == '__main__':
    main()
