"""重建S1/S2时序视图、重放全部epoch验证预测及最终行为结果；不调用GPU。"""
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
from baseline.evidence_pipeline import evidence_scores
from baseline.modernbert_binary import binary_labels, binary_metrics, calibrate_binary, predict_binary
from baseline.modernbert_capacity import replay_scores
from baseline.modernbert_data import digest, file_sha256, json_lines
from baseline.target_text_ablation import materialize
from baseline.target_text_features import build_target_inputs


def resources(path):
    limits = json.loads((path / 'resource_limits.json').read_text())
    assert json.loads((path / 'exit.json').read_text())['exit_code'] == 0
    samples = list(json_lines(path / 'resources.jsonl'))
    peak = max(x['memory.peak'] for x in samples)
    assert peak <= limits['memory.max'] <= 6 * 1024**3 and limits['memory.swap.max'] == 0
    q, period = map(int, limits['cpu.max'].split())
    assert q / period <= 2
    events = dict(line.split() for line in samples[-1]['memory.events'].splitlines())
    assert int(events['oom']) == int(events['oom_kill']) == 0
    start = dict(line.split() for line in samples[0]['cpu.stat'].splitlines())
    end = dict(line.split() for line in samples[-1]['cpu.stat'].splitlines())
    wall = samples[-1]['time'] - samples[0]['time']
    cpu = (int(end['usage_usec']) - int(start['usage_usec'])) / 1e6
    return {'peak_memory_bytes': peak, 'memory_events': events, 'wall_seconds': wall,
            'average_logical_cores': cpu / wall, 'cpu.max': limits['cpu.max'], 'memory.max': limits['memory.max']}


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
    assert config['feature_config'] == fcfg
    assert file_sha256(run / 'protocol.json') == config['protocol_sha256'] == fcfg['protocol_sha256']
    for item in [config, fcfg]:
        for name, sha in item['code_sha256'].items():
            assert file_sha256(ROOT / 'baseline' / name) == sha
    assert file_sha256(ROOT / 'artifacts/text_probe_v2/frozen/records.jsonl') == fcfg['sources']['records']
    assert file_sha256(ROOT / 'artifacts/text_probe_v2/frozen/resolved_cases.jsonl') == fcfg['sources']['cases']
    baseline = Path(config['baseline_dir'])
    assert file_sha256(baseline / 'results.json') == config['baseline_results_sha256']
    assert file_sha256(baseline / 'test_predictions.jsonl') == config['baseline_predictions_sha256']
    baseline_cfg = json.loads((baseline / 'pipeline_config.json').read_text())
    detector = Path(baseline_cfg['detector_dir'])
    for seed, sha in fcfg['predictions_sha256'].items():
        assert file_sha256(detector / f'seed-{seed}/predictions.jsonl') == sha
    rows = list(json_lines(source / 'manifest.jsonl'))
    unique = json.loads((source / 'unique_inputs.json').read_text())
    maps = {arm: {view: np.load(source / f'{arm}_{view}_mapping.npy') for view in
                  ['G', 'E_oracle', 'E_pred_42', 'E_pred_43', 'E_pred_44']} for arm in ['S1', 'S2']}
    tokenizer = AutoTokenizer.from_pretrained(fcfg['model']['local_path'], local_files_only=True)
    with tempfile.TemporaryDirectory(prefix='verify-inputs-', dir=run) as temp:
        rebuilt_rows, rebuilt_unique, sequences, rebuilt_maps = build_target_inputs(Path(temp), tokenizer,
            Path(baseline_cfg['source_dir']), detector)
        assert rebuilt_rows == rows and rebuilt_unique == unique
        for arm in maps:
            for view in maps[arm]:
                assert np.array_equal(maps[arm][view], rebuilt_maps[arm][view])
        del rebuilt_rows, rebuilt_unique, sequences, rebuilt_maps
    gc.collect()
    assert file_sha256(source / 'G_features.npy') == config['feature_sha256']
    vectors = np.load(source / 'G_features.npy', mmap_mode='r')
    assert np.load(source / 'G_done.npy').all() and np.isfinite(vectors).all()
    contract = json.loads((source / 'cache_contract.json').read_text())
    assert digest(contract) == config['feature_contract_hash']
    assert contract['manifest_hash'] == digest(rows) and contract['unique_inputs_hash'] == digest(unique)
    y = binary_labels([r['y'] for r in rows])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test'])
    result = json.loads((run / 'results.json').read_text())
    logs = list(json_lines(run / 'test_predictions.jsonl'))
    old_logs = list(json_lines(baseline / 'test_predictions.jsonl'))
    old_results = json.loads((baseline / 'results.json').read_text())
    assert [{k: v for k, v in r.items() if k != 'arm'} for r in logs if r['arm'] == 'S0'] == old_logs
    checkpoint_count, checked = 0, []
    for seed in config['training']['seeds']:
        for name in ['G', 'Oracle', 'Predicted']:
            assert result['seeds'][str(seed)]['S0'][name] == old_results['seeds'][str(seed)][name]
        for arm in ['S1', 'S2']:
            g_features = materialize(vectors, maps[arm]['G'])
            e_features = materialize(vectors, maps[arm]['E_oracle'])
            for name, features in [('G', g_features), ('E', e_features)]:
                folder = run / arm / f'{name}-seed-{seed}'
                curve = list(json_lines(folder / 'trajectory.jsonl'))
                assert [r['epoch'] for r in curve] == list(range(101))
                ET.parse(folder / 'trajectory.checked.svg')
                val_scope = np.array([i for i in val if name == 'G' or rows[i]['attack']])
                xv, yv = torch.tensor(features[val_scope]), torch.tensor(y[val_scope], dtype=torch.float32)
                saved = np.load(folder / 'validation_scores.npy')
                assert saved.shape == (101, len(val_scope)) and np.isfinite(saved).all()
                head = torch.nn.Linear(features.shape[1], 1)
                for epoch in range(101):
                    ckpt = torch.load(folder / 'checkpoints' / f'epoch-{epoch:03d}.pt', weights_only=True)
                    checkpoint_count += 1
                    assert ckpt['epoch'] == epoch and ckpt['config'] == config['training']
                    assert ckpt['feature_identity']['arm'] == arm and ckpt['feature_identity']['current_target_visible'] == (arm == 'S2')
                    head.load_state_dict(ckpt['state_dict'])
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
                for selected in [best, final]:
                    original = torch.load(folder / 'checkpoints' / f'epoch-{selected["epoch"]:03d}.pt', weights_only=True)
                    for key in selected['state_dict']:
                        assert torch.equal(selected['state_dict'][key], original['state_dict'][key])
            g = replay_scores(run / arm / f'G-seed-{seed}', g_features, rows, 'G')
            gold = replay_scores(run / arm / f'E-seed-{seed}', e_features, rows, 'E')
            pred_e = evidence_scores(run / arm / f'E-seed-{seed}/best.pt', vectors, maps[arm][f'E_pred_{seed}'])
            gold_mask = np.array([r['attack'] for r in rows])
            pred_mask = maps[arm][f'E_pred_{seed}'] >= 0
            oracle, predicted = g.copy(), g.copy()
            oracle[gold_mask] = np.maximum(g[gold_mask], gold[gold_mask])
            predicted[pred_mask] = np.maximum(g[pred_mask], pred_e[pred_mask])
            for name, scores in [('G', g), ('Oracle', oracle), ('Predicted', predicted)]:
                entry = result['seeds'][str(seed)][arm][name]
                threshold = calibrate_binary(y[val], scores[val], .05)
                assert threshold == entry['threshold_A']
                pred = predict_binary(scores[test], threshold)
                assert binary_metrics(y[test], pred) == entry['test']
                assert binary_metrics(y[val], predict_binary(scores[val], threshold)) == entry['validation']
                assert entry['validation']['A_fpr'] <= .05 + 1e-12
                saved_logs = [p for p in logs if p['arm'] == arm and p['seed'] == seed and p['configuration'] == name]
                assert [p['id'] for p in saved_logs] == [rows[i]['id'] for i in test]
                assert [p['prediction_A'] for p in saved_logs] == pred.tolist()
                assert np.allclose([p['score_A'] for p in saved_logs], scores[test], atol=1e-7)
                checked.append(f'{arm}/{name}/{seed}')
    verification = {'input_views_rebuilt_and_hash_matched': True, 'S0_unchanged': True,
                    'detector_predictions_unchanged': True, 'epoch_checkpoints_replayed': checkpoint_count,
                    'new_configurations_replayed': checked, 'test_rows_each': len(test),
                    'feature_resources': resources(source / 'run-resources'), 'training_resources': resources(run / 'run-resources'),
                    'feature_audit': json.loads((source / 'data_audit.json').read_text()),
                    'feature_elapsed_seconds': json.loads((source / 'completed.json').read_text())['elapsed_seconds'],
                    'training_elapsed_seconds': json.loads((run / 'completed.json').read_text())['elapsed_seconds'],
                    'means': result['means']}
    (run / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2))
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
