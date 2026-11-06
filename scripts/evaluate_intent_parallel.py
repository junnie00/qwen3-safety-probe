"""第三条意图分支诊断：直接OR vs 验证集联合A误报预算校准，不训练新模型。"""
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from baseline.evidence_pipeline import evidence_scores
from baseline.modernbert_binary import binary_labels, binary_metrics, predict_binary
from baseline.modernbert_capacity import replay_scores
from baseline.modernbert_data import json_lines


def main():
    if not os.environ.get('POC_CGROUP') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('需CPU-only受限wrapper')
    torch.set_num_threads(2)
    run = ROOT / 'artifacts/modernbert_oracle_poc/predicted-evidence-20261105'
    config = json.loads((run / 'pipeline_config.json').read_text())
    source, probes, detector = map(Path, [config['source_dir'], config['probe_dir'], config['detector_dir']])
    rows = list(json_lines(source / 'manifest.jsonl'))
    y = binary_labels([r['y'] for r in rows])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test'])
    g_features = np.load(source / 'G_features.npy', mmap_mode='r')
    e_features = np.load(run / 'predicted_E_cache/E_features.npy', mmap_mode='r')
    baseline_results = json.loads((run / 'results.json').read_text())
    outcome = {'scope': '三个分支均不重新训练；已有测试集上的探索性阈值对照', 'seeds': {}}
    for seed in [42, 43, 44]:
        predictions = {p['case_id']: p for p in json_lines(detector / f'seed-{seed}/predictions.jsonl')}
        intent = np.array([predictions[r['case_id']]['score_attack'] for r in rows], dtype=np.float64)
        presence = np.array([predictions[r['case_id']]['detected'] for r in rows])
        mapping = np.load(run / f'row_mapping_seed-{seed}.npy')
        g = replay_scores(probes / 'large' / f'G-seed-{seed}', g_features, rows, 'G')
        e = evidence_scores(probes / 'large' / f'E-seed-{seed}/best.pt', e_features, mapping)
        score = g.copy()
        enabled = mapping >= 0
        score[enabled] = np.maximum(g[enabled], e[enabled])
        threshold = baseline_results['seeds'][str(seed)]['Predicted']['threshold_A']
        base = predict_binary(score, threshold).astype(bool)
        assert binary_metrics(y[test], base[test].astype(int)) == baseline_results['seeds'][str(seed)]['Predicted']['test']
        direct = base | presence
        # 行为分支工作点保持不变。只调第三分支的A风险阈值，case意图阈值不改变。
        # 1.000001表示关闭独立意图告警分支，避免在val无贡献时仍对更高test分数启动。
        best = None
        for t in np.r_[1.000001, np.unique(intent[val])]:
            combined = base[val] | (intent[val] >= t)
            metrics = binary_metrics(y[val], combined.astype(int))
            key = (metrics['A_recall'], -metrics['A_fpr'], float(t))
            if metrics['A_fpr'] <= .05 + 1e-12 and (best is None or key > best[0]):
                best = (key, float(t))
        chosen = best[1]
        budgeted = base | (intent >= chosen)
        entry = {'intent_branch_risk_threshold': chosen,
                 'intent_branch_disabled': chosen > 1.,
                 'interpretation': '意图存在阈值与第三分支A风险阈值不同，不能混用'}
        for name, decision in [('two_behavior_heads', base), ('direct_OR_intent', direct), ('budgeted_OR_intent', budgeted)]:
            entry[name] = {'validation': binary_metrics(y[val], decision[val].astype(int)),
                           'test': binary_metrics(y[test], decision[test].astype(int)),
                           'new_test_A_hits': int(((decision[test] & ~base[test]) & (y[test] == 1)).sum()),
                           'new_test_false_A': int(((decision[test] & ~base[test]) & (y[test] == 0)).sum())}
        assert entry['budgeted_OR_intent']['validation']['A_fpr'] <= .05 + 1e-12
        outcome['seeds'][str(seed)] = entry
    outcome['means'] = {name: {metric: statistics.mean(x[name]['test'][metric] for x in outcome['seeds'].values())
                               for metric in ['A_recall', 'A_fpr', 'A_precision']}
                        for name in ['two_behavior_heads', 'direct_OR_intent', 'budgeted_OR_intent']}
    (run / 'intent_parallel_analysis.json').write_text(json.dumps(outcome, ensure_ascii=False, indent=2))
    print(json.dumps(outcome, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
