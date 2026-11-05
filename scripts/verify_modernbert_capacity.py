"""核验base/large对照：全部epoch验证预测重放、best/final选择、test决策与资源硬限。"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from baseline.modernbert_data import digest, file_sha256, json_lines
from baseline.modernbert_binary import calibrate_binary


def cm(y, pred):
    result = np.zeros((2, 2), dtype=int)
    for truth, decision in zip(y, pred):
        result[int(truth), int(decision)] += 1
    return result


def resources(directory):
    proof = json.loads((directory / 'resource_limits.json').read_text())
    assert json.loads((directory / 'exit.json').read_text())['exit_code'] == 0
    logs = list(json_lines(directory / 'resources.jsonl'))
    peak = max(r['memory.peak'] for r in logs)
    assert peak <= proof['memory.max'] <= 6 * 1024**3
    assert proof['memory.swap.max'] == 0
    quota, period = map(int, proof['cpu.max'].split())
    assert quota / period <= 2
    events = dict(line.split() for line in logs[-1]['memory.events'].splitlines())
    assert int(events['oom']) == int(events['oom_kill']) == 0
    start = dict(line.split() for line in logs[0]['cpu.stat'].splitlines())
    end = dict(line.split() for line in logs[-1]['cpu.stat'].splitlines())
    seconds = logs[-1]['time'] - logs[0]['time']
    cpu_seconds = (int(end['usage_usec']) - int(start['usage_usec'])) / 1e6
    return {'peak_memory_bytes': peak, 'wall_seconds': seconds, 'cpu_seconds': cpu_seconds,
            'average_logical_cores': cpu_seconds / seconds, 'memory_events': events,
            'cpu.max': proof['cpu.max'], 'memory.max': proof['memory.max']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('需CPU-only受限wrapper')
    torch.set_num_threads(2)
    run = Path(args.run_dir)
    config = json.loads((run / 'config.json').read_text())
    assert json.loads((run / 'completed.json').read_text())['config_hash'] == digest(config)
    assert config['protocol_sha256'] == file_sha256(run / 'protocol.json')
    for name, sha in config['code_sha256'].items():
        assert file_sha256(ROOT / 'baseline' / name) == sha
    features, rows = {}, None
    for model, identity in config['features'].items():
        path = Path(identity['path'])
        assert file_sha256(path / 'manifest.jsonl') == identity['manifest_sha256']
        current_rows = list(json_lines(path / 'manifest.jsonl'))
        if rows is None:
            rows = current_rows
        assert rows == current_rows and digest(rows) == config['manifest_hash']
        features[model] = {}
        for view in ['G', 'E']:
            assert file_sha256(path / f'{view}_features.npy') == identity['feature_sha256'][view]
            x = np.load(path / f'{view}_features.npy', mmap_mode='r')
            active = np.array([view == 'G' or r['attack'] for r in rows])
            assert np.isfinite(x[active]).all()
            features[model][view] = x
    y = np.array([int(r['y'] == 0) for r in rows])
    enabled = np.array([r['attack'] for r in rows])
    val_all = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test'])
    results = json.loads((run / 'results.json').read_text())
    summaries = json.loads((run / 'trajectory_summary.json').read_text())
    predictions = list(json_lines(run / 'test_predictions.jsonl'))
    verified_trajectories, verified_predictions, checkpoint_count = [], [], 0
    for model in ['base', 'large']:
        for seed in config['training']['seeds']:
            probabilities = {}
            for name in ['G', 'G_aux', 'E']:
                directory = run / model / f'{name}-seed-{seed}'
                curve = list(json_lines(directory / 'trajectory.jsonl'))
                epochs = config['training']['epochs']
                assert [r['epoch'] for r in curve] == list(range(epochs + 1))
                saved = np.load(directory / 'validation_scores.npy')
                validation_manifest = list(json_lines(directory / 'validation_manifest.jsonl'))
                val = np.array([i for i in val_all if name == 'G' or rows[i]['attack']])
                train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train' and (name == 'G' or r['attack'])])
                assert [r['id'] for r in validation_manifest] == [rows[i]['id'] for i in val]
                assert [r['y_A'] for r in validation_manifest] == y[val].tolist()
                assert saved.shape == (epochs + 1, len(val)) and np.isfinite(saved).all()
                x = features[model]['E' if name == 'E' else 'G']
                xv = torch.tensor(np.asarray(x[val]).copy())
                yv = torch.tensor(y[val], dtype=torch.float32)
                head = torch.nn.Linear(x.shape[1], 1)
                head.eval()
                for epoch in range(epochs + 1):
                    checkpoint = torch.load(directory / 'checkpoints' / f'epoch-{epoch:03d}.pt', weights_only=True)
                    checkpoint_count += 1
                    assert checkpoint['epoch'] == epoch and checkpoint['name'] == name and checkpoint['seed'] == seed
                    assert checkpoint['feature_identity'] == config['features'][model]
                    assert checkpoint['config'] == config['training']
                    assert {'optimizer_state', 'torch_rng_state', 'shuffle_rng_state'} <= checkpoint.keys()
                    assert checkpoint['train_counts'] == {'A': int(y[train].sum()), 'nonA': int((1 - y[train]).sum())}
                    head.load_state_dict(checkpoint['state_dict'])
                    with torch.inference_mode():
                        logits = head(xv).squeeze(-1)
                        score = logits.sigmoid().numpy()
                        val_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, yv))
                    assert np.allclose(score, saved[epoch], atol=1e-7)
                    assert abs(val_loss - curve[epoch]['val_bce']) < 1e-6
                    assert checkpoint['metrics'] == curve[epoch]
                summary = json.loads((directory / 'summary.json').read_text())
                best_epoch = 1 + int(np.argmin([r['val_bce'] for r in curve[1:]]))
                assert summary['best_epoch'] == best_epoch and summary['epochs_completed'] == epochs
                best = torch.load(directory / 'best.pt', weights_only=True)
                final = torch.load(directory / 'final.pt', weights_only=True)
                assert best['epoch'] == best_epoch and final['epoch'] == epochs
                best_original = torch.load(directory / 'checkpoints' / f'epoch-{best_epoch:03d}.pt', weights_only=True)
                final_original = torch.load(directory / 'checkpoints' / f'epoch-{epochs:03d}.pt', weights_only=True)
                for key in best['state_dict']:
                    assert torch.equal(best['state_dict'][key], best_original['state_dict'][key])
                    assert torch.equal(final['state_dict'][key], final_original['state_dict'][key])
                head.load_state_dict(best['state_dict'])
                scores = np.full(len(rows), np.nan)
                valid = np.array([i for i, r in enumerate(rows) if name != 'E' or r['attack']])
                with torch.inference_mode():
                    for offset in range(0, len(valid), 256):
                        idx = valid[offset:offset + 256]
                        scores[idx] = head(torch.tensor(np.asarray(x[idx]).copy())).squeeze(-1).sigmoid().numpy()
                probabilities[name] = scores
                verified_trajectories.append(f'{model}/{name}/{seed}')
            for scheme in ['M0', 'M1', 'M2']:
                entry = results['seeds'][str(seed)][model][scheme]
                scores = probabilities['G'].copy()
                if scheme != 'M0':
                    aux = probabilities['G_aux' if scheme == 'M1' else 'E']
                    scores[enabled] = np.maximum(scores[enabled], aux[enabled])
                # 用独立穷举实现核验训练阶段的快速排序校准。
                threshold = calibrate_binary(y[val_all], scores[val_all], .05)
                assert threshold == entry['threshold_A']
                pred = (scores >= threshold).astype(int)
                assert cm(y[val_all], pred[val_all]).tolist() == entry['validation']['confusion']
                assert entry['validation']['A_fpr'] <= .05 + 1e-12
                assert cm(y[test], pred[test]).tolist() == entry['test']['confusion']
                logs = [r for r in predictions if r['model'] == model and r['seed'] == seed and r['configuration'] == scheme]
                assert [r['id'] for r in logs] == [rows[i]['id'] for i in test]
                assert [r['prediction_A'] for r in logs] == pred[test].tolist()
                assert np.allclose([r['score_A'] for r in logs], scores[test], atol=1e-7)
                verified_predictions.append(f'{model}/{scheme}/{seed}')
    large_path = Path(config['features']['large']['path'])
    benchmark = json.loads((large_path / 'batch_benchmark.json').read_text())
    large_completed = json.loads((large_path / 'completed.json').read_text())
    assert large_completed['trainable_backbone_parameters'] == 0
    verification = {'verified_trajectories': verified_trajectories, 'epoch_checkpoints_replayed': checkpoint_count,
                    'verified_test_configurations': verified_predictions, 'test_rows_each': len(test),
                    'features_have_identical_token_inputs': True,
                    'large_feature_resources': resources(large_path / 'run-resources'),
                    'head_training_resources': resources(run / 'run-resources'),
                    'large_parameters': large_completed['parameters'],
                    'large_hidden_size': large_completed['hidden_size'],
                    'large_extraction_seconds': large_completed['elapsed_seconds'],
                    'selected_batches': benchmark['selected_batches'],
                    'large_cuda_allocator_peak_bytes': max(r.get('cuda_peak_allocated', 0) for r in benchmark['measurements']),
                    'large_cuda_reserved_peak_bytes': max(r.get('cuda_peak_reserved', 0) for r in benchmark['measurements']),
                    'best_epochs': {model: {f'{r["name"]}/{r["seed"]}': r['best_epoch'] for r in values}
                                    for model, values in summaries.items()},
                    'means': results['means']}
    (run / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2))
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
