"""独立核验预测证据：检测头重放、原文合法性、gold-free输入重建、行为预测与资源。"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from transformers import AutoTokenizer
from baseline.evidence_detector import case_indices, detector_metrics, select_predictions
from baseline.evidence_pipeline import build_inputs, evidence_scores
from baseline.modernbert_binary import binary_labels, binary_metrics, calibrate_binary
from baseline.modernbert_capacity import replay_scores
from baseline.modernbert_data import digest, file_sha256, json_lines


def resource_check(path):
    proof = json.loads((path / 'resource_limits.json').read_text())
    assert json.loads((path / 'exit.json').read_text())['exit_code'] == 0
    logs = list(json_lines(path / 'resources.jsonl'))
    peak = max(x['memory.peak'] for x in logs)
    assert peak <= proof['memory.max'] <= 6 * 1024**3 and proof['memory.swap.max'] == 0
    quota, period = map(int, proof['cpu.max'].split())
    assert quota / period <= 2
    events = dict(line.split() for line in logs[-1]['memory.events'].splitlines())
    assert int(events['oom']) == 0 and int(events['oom_kill']) == 0
    return {'peak_memory_bytes': peak, 'wall_seconds': logs[-1]['time'] - logs[0]['time'],
            'memory_events': events, 'cpu.max': proof['cpu.max'], 'memory.max': proof['memory.max']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('需CPU-only受限wrapper')
    torch.set_num_threads(2)
    run = Path(args.run_dir)
    config = json.loads((run / 'pipeline_config.json').read_text())
    assert json.loads((run / 'completed.json').read_text())['config_hash'] == digest(config)
    assert file_sha256(run / 'protocol.json') == config['protocol_sha256']
    for name, sha in config['code_sha256'].items():
        assert file_sha256(ROOT / 'baseline' / name) == sha
    detector = Path(config['detector_dir'])
    dcfg = json.loads((detector / 'config.json').read_text())
    assert json.loads((detector / 'completed.json').read_text())['config_hash'] == digest(dcfg)
    for name, sha in dcfg['code_sha256'].items():
        assert file_sha256(ROOT / 'baseline' / name) == sha
    cases = list(json_lines(ROOT / 'artifacts/text_probe_v2/frozen/resolved_cases.jsonl'))
    by_id = {c['id']: c for c in cases}
    candidates = list(json_lines(detector / 'candidates.jsonl'))
    features = np.load(detector / 'G_features.npy', mmap_mode='r')
    for candidate in candidates:
        case = by_id[candidate['case_id']]
        msg = case['context']['ordered_messages'][candidate['message_index']]
        assert msg['role'] == 'tool' and candidate['text'] == msg['content'][candidate['start']:candidate['end']]
    groups = case_indices(cases, candidates)
    val_case = np.array([i for i, c in enumerate(cases) if c['split'] == 'val'])
    val_windows = np.array([i for i, c in enumerate(candidates) if c['split'] == 'val'])
    relative = {int(i): j for j, i in enumerate(val_windows)}
    val_groups = [np.array([relative[int(i)] for i in groups[c]]) for c in val_case]
    case_y = np.array([int(bool(c.get('attack_str'))) for c in cases])
    x_val = torch.tensor(np.asarray(features[val_windows]).copy())
    dresults = json.loads((detector / 'results.json').read_text())
    predictions, checkpoint_count = {}, 0
    for seed in [42, 43, 44]:
        directory = detector / f'seed-{seed}'
        assert file_sha256(directory / 'predictions.jsonl') == config['detector_predictions_sha256'][str(seed)]
        curve = list(json_lines(directory / 'trajectory.jsonl'))
        assert [r['epoch'] for r in curve] == list(range(1, 101))
        head = torch.nn.Linear(features.shape[1], 1)
        head.eval()
        for epoch, record in enumerate(curve, 1):
            ckpt = torch.load(directory / 'checkpoints' / f'epoch-{epoch:03d}.pt', weights_only=True)
            checkpoint_count += 1
            assert ckpt['epoch'] == epoch and ckpt['label_mapping'] == {'no_attack': 0, 'attack': 1}
            head.load_state_dict(ckpt['state_dict'])
            with torch.inference_mode():
                logits = head(x_val).squeeze(-1)
                maxima = torch.stack([logits[idx].max() for idx in val_groups])
                loss = float(torch.nn.functional.binary_cross_entropy_with_logits(maxima, torch.tensor(case_y[val_case], dtype=torch.float32)))
            assert abs(loss - record['val_case_bce']) < 1e-6
        best = torch.load(directory / 'best.pt', weights_only=True)
        assert best['epoch'] == 1 + int(np.argmin([r['val_case_bce'] for r in curve]))
        head.load_state_dict(best['state_dict'])
        scores = np.empty(len(candidates), dtype=np.float32)
        with torch.inference_mode():
            for offset in range(0, len(candidates), 256):
                scores[offset:offset + 256] = head(torch.tensor(np.asarray(features[offset:offset + 256]).copy())).squeeze(-1).sigmoid().numpy()
        assert np.allclose(scores, np.load(directory / 'candidate_scores.npy'), atol=1e-7)
        threshold = calibrate_binary(case_y[val_case], np.array([scores[groups[i]].max() for i in val_case]), .05)
        assert threshold == dresults[str(seed)]['threshold']
        # 推理仅给ID，不给attack_str或行为标签。
        pred = select_predictions([{'id': c['id']} for c in cases], candidates, scores, threshold)
        assert pred == list(json_lines(directory / 'predictions.jsonl'))
        assert detector_metrics(cases, pred) == dresults[str(seed)]['metrics']
        predictions[seed] = {p['case_id']: p for p in pred}
    source, probes = Path(config['source_dir']), Path(config['probe_dir'])
    rows = list(json_lines(source / 'manifest.jsonl'))
    assert file_sha256(source / 'manifest.jsonl') == config['source_manifest_sha256']
    for view, sha in config['source_features_sha256'].items():
        assert file_sha256(source / f'{view}_features.npy') == sha
    for name, sha in config['probe_checkpoints_sha256'].items():
        head_name, seed = name.split('/')
        assert file_sha256(probes / 'large' / f'{head_name}-seed-{seed}/best.pt') == sha
    tokenizer = AutoTokenizer.from_pretrained(config['model']['local_path'], local_files_only=True)
    # 删除gold/evaluator/context等所有无关字段，重建输入及映射，证明无gold回填。
    public_cases = {c['id']: {'id': c['id'], 'question': c['question']} for c in cases}
    unique, _, maps = build_inputs(rows, public_cases, predictions, tokenizer)
    assert unique == json.loads((run / 'predicted_inputs.json').read_text())
    for seed, mapping in maps.items():
        assert np.array_equal(mapping, np.load(run / f'row_mapping_seed-{seed}.npy'))
    vectors = np.load(run / 'predicted_E_cache/E_features.npy', mmap_mode='r')
    assert np.isfinite(vectors).all()
    assert np.load(run / 'predicted_E_cache/E_done.npy').all()
    source_g, source_e = [np.load(source / f'{v}_features.npy', mmap_mode='r') for v in ['G', 'E']]
    previous_inputs = {r['E_hash']: i for i, r in enumerate(rows) if r.get('E_hash')}
    cache_hits = 0
    for i, row in enumerate(unique):
        if row['E_hash'] in previous_inputs:
            assert np.array_equal(vectors[i], source_e[previous_inputs[row['E_hash']]])
            cache_hits += 1
    result = json.loads((run / 'results.json').read_text())
    assert result['encoding']['exact_token_cache_hits'] == cache_hits
    y = binary_labels([r['y'] for r in rows])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test'])
    original = json.loads((probes / 'results.json').read_text())
    logs = list(json_lines(run / 'test_predictions.jsonl'))
    verified = []
    for seed in predictions:
        g = replay_scores(probes / 'large' / f'G-seed-{seed}', source_g, rows, 'G')
        e_gold = replay_scores(probes / 'large' / f'E-seed-{seed}', source_e, rows, 'E')
        e_pred = evidence_scores(probes / 'large' / f'E-seed-{seed}/best.pt', vectors, maps[seed])
        enabled = maps[seed] >= 0
        golden = np.array([r['attack'] for r in rows])
        oracle, predicted = g.copy(), g.copy()
        oracle[golden] = np.maximum(g[golden], e_gold[golden])
        predicted[enabled] = np.maximum(g[enabled], e_pred[enabled])
        for name, scores in [('G', g), ('Oracle', oracle), ('Predicted', predicted)]:
            entry = result['seeds'][str(seed)][name]
            threshold = (calibrate_binary(y[val], scores[val], .05) if name == 'Predicted' else
                         original['seeds'][str(seed)]['large']['M0' if name == 'G' else 'M2']['threshold_A'])
            assert threshold == entry['threshold_A']
            pred = (scores.astype(np.float64) >= threshold).astype(int)
            assert binary_metrics(y[test], pred[test]) == entry['test']
            assert binary_metrics(y[val], pred[val]) == entry['validation']
            assert entry['validation']['A_fpr'] <= .05 + 1e-12
            saved = [p for p in logs if p['seed'] == seed and p['configuration'] == name]
            assert [p['id'] for p in saved] == [rows[i]['id'] for i in test]
            assert [p['prediction_A'] for p in saved] == pred[test].tolist()
            assert np.allclose([p['score_A'] for p in saved], scores[test], atol=1e-7)
            verified.append(f'{seed}/{name}')
    means = {metric: statistics.mean(r['metrics']['test'][metric] for r in dresults.values())
             for metric in ['attack_recall', 'clean_fpr', 'full_gold_coverage_rate', 'exact_evidence_rate']}
    verification = {'detector_epoch_checkpoints_replayed': checkpoint_count, 'detector_cases_each_seed': len(cases),
                    'gold_free_inputs_rebuilt': True, 'exact_cache_hits_verified': cache_hits,
                    'behavior_configurations_verified': verified, 'test_blocks_each': len(test),
                    'detector_test_means': means, 'behavior_test_means': result['means'],
                    'detector_resources': resource_check(detector / 'run-resources'),
                    'pipeline_resources': resource_check(run / 'run-resources')}
    (run / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2))
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    for seed, entry in result['seeds'].items():
        print('seed', seed, 'new_misses', entry['oracle_hit_predicted_miss']['counts'],
              'bootstrap', entry['bootstrap']['percentile_95'])


if __name__ == '__main__':
    main()
