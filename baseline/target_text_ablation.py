"""S0预测基线、S1加入主模型历史、S2发布前审查的受控比较。"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from baseline.evidence_pipeline import evidence_scores
from baseline.modernbert_binary import binary_labels, binary_metrics, predict_binary
from baseline.modernbert_capacity import TRAIN_CONFIG, replay_scores
from baseline.modernbert_data import digest, file_sha256, json_lines
from baseline.modernbert_poc import write_json
from baseline.modernbert_trajectory import fast_calibrate, train_trajectory


def materialize(vectors, mapping):
    output = np.full((len(mapping), vectors.shape[1]), np.nan, dtype=np.float32)
    valid = mapping >= 0
    output[valid] = vectors[mapping[valid]]
    return output


def paired_intervals(y, predictions, groups, seed, iterations=500):
    unique = sorted(set(groups))
    matrices = {arm: np.stack([np.bincount(y[groups == g] * 2 + pred[groups == g], minlength=4).reshape(2, 2)
                              for g in unique]) for arm, pred in predictions.items()}
    pairs = [('S1', 'S0'), ('S2', 'S0'), ('S2', 'S1')]
    values = {f'{a}-minus-{b}/{metric}': [] for a, b in pairs for metric in ['A_recall', 'A_fpr']}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        selection = rng.integers(0, len(unique), len(unique))
        stats = {}
        for arm, matrices_arm in matrices.items():
            tn, fp, fn, tp = matrices_arm[selection].sum(axis=0).ravel()
            stats[arm] = {'A_recall': tp / (tp + fn) if tp + fn else np.nan,
                          'A_fpr': fp / (fp + tn) if fp + tn else np.nan}
        for a, b in pairs:
            for metric in ['A_recall', 'A_fpr']:
                values[f'{a}-minus-{b}/{metric}'].append(stats[a][metric] - stats[b][metric])
    return {'unit': 'source_group', 'groups': len(unique), 'iterations': iterations,
            'percentile_95': {k: np.nanquantile(v, [.025, .975]).tolist() for k, v in values.items()}}


def evaluate(out, rows, vectors, maps, baseline):
    baseline_results = json.loads((baseline / 'results.json').read_text())
    baseline_predictions = list(json_lines(baseline / 'test_predictions.jsonl'))
    mechanisms = {r['id']: r['mechanism'] for r in json_lines('artifacts/modernbert_oracle_poc/attack-recall-analysis-20261105/A_cases.jsonl')}
    y = binary_labels([r['y'] for r in rows])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test'])
    gold_enabled = np.array([r['attack'] for r in rows])
    results = {'seeds': {}, 'bootstrap': {}, 'arms': {'S0': '无主模型文本/提前预测', 'S1': '过去主模型文本/提前预测',
                                                    'S2': '当前主模型输出可见/发布前审查'}}
    with (out / 'test_predictions.jsonl').open('w') as f:
        for seed in TRAIN_CONFIG['seeds']:
            entries = {'S0': {name: baseline_results['seeds'][str(seed)][name] for name in ['G', 'Oracle', 'Predicted']}}
            old_pred = [r for r in baseline_predictions if r['seed'] == seed and r['configuration'] == 'Predicted']
            assert [r['id'] for r in old_pred] == [rows[i]['id'] for i in test]
            decisions = {'S0': np.array([r['prediction_A'] for r in old_pred])}
            for record in baseline_predictions:
                if record['seed'] == seed:
                    f.write(json.dumps({'arm': 'S0', **record}) + '\n')
            for arm in ['S1', 'S2']:
                xg = materialize(vectors, maps[arm]['G'])
                xe = materialize(vectors, maps[arm]['E_oracle'])
                g = replay_scores(out / arm / f'G-seed-{seed}', xg, rows, 'G')
                oracle_e = replay_scores(out / arm / f'E-seed-{seed}', xe, rows, 'E')
                del xg, xe
                mapping = maps[arm][f'E_pred_{seed}']
                predicted_e = evidence_scores(out / arm / f'E-seed-{seed}/best.pt', vectors, mapping)
                oracle, predicted = g.copy(), g.copy()
                oracle[gold_enabled] = np.maximum(g[gold_enabled], oracle_e[gold_enabled])
                enabled = mapping >= 0
                predicted[enabled] = np.maximum(g[enabled], predicted_e[enabled])
                scores = {'G': g, 'Oracle': oracle, 'Predicted': predicted}
                entries[arm] = {}
                for name, score in scores.items():
                    threshold = fast_calibrate(y[val], score[val], .05)
                    pred = predict_binary(score[test], threshold)
                    strata = {}
                    for field in ['domain', 'label_source', 'variant']:
                        labels = np.array([rows[i][field] for i in test])
                        strata[field] = {str(k): binary_metrics(y[test][labels == k], pred[labels == k]) for k in sorted(set(labels))}
                    mechanism_stats = {}
                    for mechanism in sorted(set(mechanisms.values())):
                        selected = np.array([j for j, i in enumerate(test) if mechanisms.get(rows[i]['id']) == mechanism])
                        if len(selected):
                            mechanism_stats[mechanism] = {'A_blocks': len(selected), 'A_recall': float(pred[selected].mean())}
                    entries[arm][name] = {'threshold_A': threshold, 'validation': binary_metrics(y[val], predict_binary(score[val], threshold)),
                                         'test': binary_metrics(y[test], pred), 'strata': strata, 'A_mechanisms': mechanism_stats}
                    if name == 'Predicted':
                        decisions[arm] = pred
                    for j, i in enumerate(test):
                        row = rows[i]
                        f.write(json.dumps({'arm': arm, 'seed': seed, 'configuration': name, 'id': row['id'],
                                            'case_id': row['case_id'], 'group_id': row['group_id'], 'y_A': int(y[i]),
                                            'prediction_A': int(pred[j]), 'score_A': float(score[i]),
                                            'detector_positive': bool(enabled[i]),
                                            'current_target_visible': arm == 'S2'}) + '\n')
            results['seeds'][str(seed)] = entries
            results['bootstrap'][str(seed)] = paired_intervals(y[test], decisions, np.array([rows[i]['group_id'] for i in test]), seed)
            f.flush()
    results['means'] = {arm: {name: {metric: statistics.mean(e[arm][name]['test'][metric] for e in results['seeds'].values())
                                   for metric in ['A_recall', 'A_fpr', 'A_precision', 'A_f1']}
                             for name in ['G', 'Oracle', 'Predicted']} for arm in ['S0', 'S1', 'S2']}
    write_json(out / 'results.json', results)
    return results


def render_report(out, results, summaries, elapsed):
    lines = ['# 主模型文本输入消融：预测与发布前审查', '',
             'S0是现有无主模型文本的预测证据系统；S1加入过去主模型输出；S2再加入当前待发布输出。',
             'S2只能解释为生成后的审查，不是提前预测效果，不证明在线发布/工具执行已被安全阻断。', '',
             '## 主要结果：实际预测证据，三个seed均值', '',
             '| 方案 | A召回 | A误报率 | A精确率 | A-F1 |', '| --- | ---: | ---: | ---: | ---: |']
    for arm in ['S0', 'S1', 'S2']:
        r = results['means'][arm]['Predicted']
        lines.append(f'| {arm} | {r["A_recall"]:.2%} | {r["A_fpr"]:.2%} | {r["A_precision"]:.2%} | {r["A_f1"]:.4f} |')
    lines += ['', '## 单通用头与oracle证据诊断', '', '| 方案 | G A召回 | G A误报 | Oracle A召回 | Oracle A误报 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for arm in ['S0', 'S1', 'S2']:
        g, e = results['means'][arm]['G'], results['means'][arm]['Oracle']
        lines.append(f'| {arm} | {g["A_recall"]:.2%} | {g["A_fpr"]:.2%} | {e["A_recall"]:.2%} | {e["A_fpr"]:.2%} |')
    lines += ['', '## 新训练头的轨迹', '', '| 方案 | 头 | seed | 最佳epoch | 最佳val BCE | 100轮val BCE | 最佳点在最后20轮 |',
              '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for arm, values in summaries.items():
        for s in values:
            lines.append(f'| {arm} | {s["name"]} | {s["seed"]} | {s["best_epoch"]} | {s["best_val_bce"]:.4f} | '
                         f'{s["final_val_bce"]:.4f} | {s["best_in_last20"]} |')
    lines += ['', '## 对照边界', '',
              '- 不新增样本，仍是相同4110/1078/1883条有效块和来源组划分。意图检测器/原文片段/启用mask均冻结。',
              '- 骨干保持冻结ModernBERT-large FP32与末层mean pooling；当前草稿、TASK、聊天选择和证据规则不变。',
              '- 历史只来自input.history_pairs各条target，拼接后取最近256token；不读取历史draft、接受计数或标签。',
              '- S1不读当前supervision，S2只读target_output，不读target_text_so_far、参考答案、判定说明和未来输出。',
              '- 与S0一致，E头仍在已知攻击子集上用oracle原文训练/选点；主结果评估统一改用冻结检测器的预测证据。未同时改成预测证据训练，避免引入第二个变量。',
              '- 新头全部跑满100epoch，仅按未加权val BCE选择best；final为100轮。每轮权重、优化器、RNG、验证预测及CSV/SVG曲线全部保存。',
              '- S0复用原已核验头与阈值；S1/S2按相同的val A误报≤5%预算重新校准，不在test上调阈值。',
              '- 输入上限4096；历史字段占用预算，只允许按原策略裁剪G聊天材料。实际裁剪统计见features/data_audit.json。',
              '- 时序审计核对全16196条记录：历史等于初始snapshot加严格此前各块target。初始snapshot可能来自此前对话的主模型输出。',
              '- 相同test已多次查看，结果仅探索性；当前文本可见带来的增益不可归因为预测器自身更强。',
              '- 接入线上前仍需实现输出缓冲，并在文本发布/工具执行前运行S2审查；已发生的外部副作用不能靠事后检测撤回。',
              f'- 头训练与评估逻辑耗时{elapsed:.2f}秒，GPU特征提取耗时另计。', '']
    (out / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('行为头需CPU-only受限wrapper')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out = Path(args.run_dir)
    source = out / 'features'
    baseline = Path('artifacts/modernbert_oracle_poc/predicted-evidence-20261105')
    assert json.loads((source / 'run-resources/exit.json').read_text())['exit_code'] == 0
    assert (baseline / 'verification.json').exists()
    if (out / 'config.json').exists():
        raise ValueError('不可覆盖既有训练')
    start = time.perf_counter()
    feature_config = json.loads((source / 'config.json').read_text())
    assert json.loads((source / 'completed.json').read_text())['config_hash'] == digest(feature_config)
    baseline_config = json.loads((baseline / 'pipeline_config.json').read_text())
    assert feature_config['predictions_sha256'] == baseline_config['detector_predictions_sha256']
    rows = list(json_lines(source / 'manifest.jsonl'))
    original_rows = list(json_lines(Path(baseline_config['source_dir']) / 'manifest.jsonl'))
    assert rows == original_rows
    contract = json.loads((source / 'cache_contract.json').read_text())
    assert contract['manifest_hash'] == digest(rows)
    vectors = np.load(source / 'G_features.npy', mmap_mode='r')
    assert np.load(source / 'G_done.npy').all() and np.isfinite(vectors).all()
    maps = {arm: {view: np.load(source / f'{arm}_{view}_mapping.npy') for view in
                  ['G', 'E_oracle', 'E_pred_42', 'E_pred_43', 'E_pred_44']} for arm in ['S1', 'S2']}
    config = {'training': TRAIN_CONFIG, 'feature_config': feature_config, 'feature_sha256': file_sha256(source / 'G_features.npy'),
              'feature_contract_hash': digest(contract), 'manifest_hash': digest(rows),
              'baseline_dir': str(baseline.resolve()), 'baseline_results_sha256': file_sha256(baseline / 'results.json'),
              'baseline_predictions_sha256': file_sha256(baseline / 'test_predictions.jsonl'),
              'protocol_sha256': file_sha256(out / 'protocol.json'),
              'code_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in
                              ['target_text_ablation.py', 'target_text_features.py', 'modernbert_trajectory.py', 'evidence_pipeline.py']}}
    write_json(out / 'config.json', config)
    summaries = {}
    for arm in ['S1', 'S2']:
        summaries[arm] = []
        g = materialize(vectors, maps[arm]['G'])
        e = materialize(vectors, maps[arm]['E_oracle'])
        for seed in TRAIN_CONFIG['seeds']:
            for name, features in [('G', g), ('E', e)]:
                identity = {'arm': arm, 'view': name, 'feature_sha256': config['feature_sha256'],
                            'mapping_sha256': file_sha256(source / f'{arm}_{"G" if name == "G" else "E_oracle"}_mapping.npy'),
                            'current_target_visible': arm == 'S2'}
                summaries[arm].append(train_trajectory(name, features, rows, seed, out / arm / f'{name}-seed-{seed}', TRAIN_CONFIG, identity))
        del g, e
    write_json(out / 'trajectory_summary.json', summaries)
    results = evaluate(out, rows, vectors, maps, baseline)
    elapsed = time.perf_counter() - start
    render_report(out, results, summaries, elapsed)
    write_json(out / 'completed.json', {'config_hash': digest(config), 'elapsed_seconds': elapsed})
    print(f'S0/S1/S2对照完成：{out / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
