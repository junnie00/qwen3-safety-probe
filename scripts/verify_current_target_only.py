"""S3复核：重建无额外历史输入、逐epoch重放、冻结S0/S2对照和资源检查。"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from transformers import AutoTokenizer
from baseline.current_target_only import REFERENCE, VIEWS, build_inputs
from baseline.evidence_pipeline import evidence_scores
from baseline.modernbert_binary import binary_labels, binary_metrics, calibrate_binary, predict_binary
from baseline.modernbert_capacity import replay_scores
from baseline.modernbert_data import digest, file_sha256, json_lines
from baseline.target_text_ablation import materialize


def resource_summary(path):
    limits = json.loads((path / 'resource_limits.json').read_text())
    assert json.loads((path / 'exit.json').read_text())['exit_code'] == 0
    samples = list(json_lines(path / 'resources.jsonl'))
    peak = max(x['memory.peak'] for x in samples)
    assert peak <= limits['memory.max'] <= 6 * 1024**3 and limits['memory.swap.max'] == 0
    q, period = map(int, limits['cpu.max'].split())
    assert q / period <= 2
    events = dict(line.split() for line in samples[-1]['memory.events'].splitlines())
    assert int(events['oom']) == int(events['oom_kill']) == 0
    first = dict(line.split() for line in samples[0]['cpu.stat'].splitlines())
    last = dict(line.split() for line in samples[-1]['cpu.stat'].splitlines())
    wall = samples[-1]['time'] - samples[0]['time']
    cpu = (int(last['usage_usec']) - int(first['usage_usec'])) / 1e6
    return {'peak_memory_bytes': peak, 'wall_seconds': wall, 'average_logical_cores': cpu / wall,
            'memory_events': events, 'cpu.max': limits['cpu.max'], 'memory.max': limits['memory.max']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('需CPU-only受限wrapper')
    torch.set_num_threads(2)
    run = Path(args.run_dir)
    source = run / 'features'
    config = json.loads((run / 'config.json').read_text())
    fcfg = json.loads((source / 'config.json').read_text())
    assert json.loads((run / 'completed.json').read_text())['config_hash'] == digest(config)
    assert json.loads((source / 'completed.json').read_text())['config_hash'] == digest(fcfg)
    assert fcfg == config['feature_config']
    assert file_sha256(run / 'protocol.json') == config['protocol_sha256'] == fcfg['protocol_sha256']
    for data in [config, fcfg]:
        for name, sha in data['code_sha256'].items():
            assert file_sha256(ROOT / name) == sha
    assert file_sha256(ROOT / 'artifacts/text_probe_v2/frozen/records.jsonl') == fcfg['sources']['records']
    assert file_sha256(ROOT / 'artifacts/text_probe_v2/frozen/resolved_cases.jsonl') == fcfg['sources']['cases']
    assert file_sha256(REFERENCE / 'results.json') == config['reference_results_sha256']
    assert file_sha256(REFERENCE / 'test_predictions.jsonl') == config['reference_predictions_sha256']
    rows = list(json_lines(source / 'manifest.jsonl'))
    assert rows == list(json_lines(REFERENCE / 'features/manifest.jsonl'))
    assert digest(rows) == config['manifest_hash']
    unique = json.loads((source / 'unique_inputs.json').read_text())
    maps = {v: np.load(source / f'{v}_mapping.npy') for v in VIEWS}
    tokenizer = AutoTokenizer.from_pretrained(fcfg['model']['local_path'], local_files_only=True)
    with tempfile.TemporaryDirectory(prefix='verify-inputs-', dir=run) as temp:
        rr, ru, sequences, rm = build_inputs(Path(temp), tokenizer)
        assert rr == rows and ru == unique
        for view in VIEWS:
            assert np.array_equal(rm[view], maps[view])
        del rr, ru, sequences, rm
    gc.collect()
    assert file_sha256(source / 'G_features.npy') == config['feature_sha256']
    vectors = np.load(source / 'G_features.npy', mmap_mode='r')
    assert np.load(source / 'G_done.npy').all() and np.isfinite(vectors).all()
    contract = json.loads((source / 'cache_contract.json').read_text())
    assert contract['unique_inputs_hash'] == digest(unique) and contract['manifest_hash'] == digest(rows)
    result = json.loads((run / 'results.json').read_text())
    previous = json.loads((REFERENCE / 'results.json').read_text())
    logs = list(json_lines(run / 'test_predictions.jsonl'))
    old_logs = list(json_lines(REFERENCE / 'test_predictions.jsonl'))
    for arm in ['S0', 'S2']:
        assert [r for r in logs if r['arm'] == arm] == [r for r in old_logs if r['arm'] == arm]
        for seed in ['42', '43', '44']:
            assert result['seeds'][seed][arm] == previous['seeds'][seed][arm]
    y = binary_labels([r['y'] for r in rows])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test'])
    g_features, e_features = materialize(vectors, maps['G']), materialize(vectors, maps['E_oracle'])
    replayed, configurations = 0, []
    for seed in config['training']['seeds']:
        for name, features in [('G', g_features), ('E', e_features)]:
            folder = run / 'S3' / f'{name}-seed-{seed}'
            curve = list(json_lines(folder / 'trajectory.jsonl'))
            assert [r['epoch'] for r in curve] == list(range(101))
            ET.parse(folder / 'trajectory.checked.svg')
            val_scope = np.array([i for i in val if name == 'G' or rows[i]['attack']])
            saved = np.load(folder / 'validation_scores.npy')
            assert saved.shape == (101, len(val_scope)) and np.isfinite(saved).all()
            xv, yv = torch.tensor(features[val_scope]), torch.tensor(y[val_scope], dtype=torch.float32)
            head = torch.nn.Linear(features.shape[1], 1)
            for epoch in range(101):
                checkpoint = torch.load(folder / 'checkpoints' / f'epoch-{epoch:03d}.pt', weights_only=True)
                replayed += 1
                assert checkpoint['epoch'] == epoch and checkpoint['config'] == config['training']
                assert checkpoint['feature_identity']['arm'] == 'S3'
                assert checkpoint['feature_identity']['current_target_visible'] is True
                assert checkpoint['feature_identity']['extra_target_history_visible'] is False
                head.load_state_dict(checkpoint['state_dict'])
                with torch.inference_mode():
                    logits = head(xv).squeeze(-1)
                    score = logits.sigmoid().numpy()
                    loss = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, yv))
                assert np.allclose(score, saved[epoch], atol=1e-7)
                assert abs(loss - curve[epoch]['val_bce']) < 1e-6
            best = torch.load(folder / 'best.pt', weights_only=True)
            final = torch.load(folder / 'final.pt', weights_only=True)
            assert best['epoch'] == 1 + int(np.argmin([r['val_bce'] for r in curve[1:]]))
            assert final['epoch'] == 100
            for ckpt in [best, final]:
                original = torch.load(folder / 'checkpoints' / f'epoch-{ckpt["epoch"]:03d}.pt', weights_only=True)
                for key in ckpt['state_dict']:
                    assert torch.equal(ckpt['state_dict'][key], original['state_dict'][key])
        g = replay_scores(run / 'S3' / f'G-seed-{seed}', g_features, rows, 'G')
        e = replay_scores(run / 'S3' / f'E-seed-{seed}', e_features, rows, 'E')
        pred_e = evidence_scores(run / 'S3' / f'E-seed-{seed}/best.pt', vectors, maps[f'E_pred_{seed}'])
        gold_mask = np.array([r['attack'] for r in rows])
        pred_mask = maps[f'E_pred_{seed}'] >= 0
        oracle, predicted = g.copy(), g.copy()
        oracle[gold_mask] = np.maximum(g[gold_mask], e[gold_mask])
        predicted[pred_mask] = np.maximum(g[pred_mask], pred_e[pred_mask])
        for name, scores in [('G', g), ('Oracle', oracle), ('Predicted', predicted)]:
            entry = result['seeds'][str(seed)]['S3'][name]
            threshold = calibrate_binary(y[val], scores[val], .05)
            assert threshold == entry['threshold_A']
            assert entry['validation']['A_fpr'] <= .05 + 1e-12
            pred = predict_binary(scores[test], threshold)
            assert binary_metrics(y[test], pred) == entry['test']
            assert binary_metrics(y[val], predict_binary(scores[val], threshold)) == entry['validation']
            saved_logs = [r for r in logs if r['arm'] == 'S3' and r['seed'] == seed and r['configuration'] == name]
            assert [r['id'] for r in saved_logs] == [rows[i]['id'] for i in test]
            assert [r['prediction_A'] for r in saved_logs] == pred.tolist()
            assert np.allclose([r['score_A'] for r in saved_logs], scores[test], atol=1e-7)
            configurations.append(f'S3/{name}/{seed}')
    report = {'input_views_rebuilt': True, 'S0_S2_unchanged': True, 'extra_history_field_included': False,
        'epoch_checkpoints_replayed': replayed, 'configurations_replayed': configurations,
        'test_blocks_each': len(test), 'feature_resources': resource_summary(source / 'run-resources'),
        'training_resources': resource_summary(run / 'run-resources'),
        'input_audit': json.loads((source / 'data_audit.json').read_text()),
        'feature_elapsed_seconds': json.loads((source / 'completed.json').read_text())['elapsed_seconds'],
        'training_elapsed_seconds': json.loads((run / 'completed.json').read_text())['elapsed_seconds'],
        'means': result['means']}
    (run / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
