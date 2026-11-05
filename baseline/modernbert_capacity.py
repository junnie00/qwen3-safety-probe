"""ModernBERT base/large 冻结特征、二分类头跑满100epoch的配对比较。"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from baseline.modernbert_data import digest, file_sha256, json_lines
from baseline.modernbert_binary import aggregate, binary_labels, binary_metrics, predict_binary
from baseline.modernbert_trajectory import fast_calibrate, train_trajectory

TRAIN_CONFIG = {'lr': 1e-3, 'weight_decay': 1e-4, 'head_batch': 128, 'epochs': 100,
                'diagnostic_patience': 10, 'early_termination': False, 'max_a_fpr': .05,
                'seeds': [42, 43, 44], 'checkpoint_selection': 'minimum_unweighted_val_BCE_epochs_1_to_100',
                'loss': 'BCEWithLogitsLoss(pos_weight=train_nonA/train_A)'}


def load_features(path):
    path = Path(path)
    if (path / 'INVALID.json').exists():
        raise ValueError('无效特征源')
    config = json.loads((path / 'config.json').read_text())
    contract = json.loads((path / 'cache_contract.json').read_text())
    rows = list(json_lines(path / 'manifest.jsonl'))
    assert config == contract['config'] and digest(rows) == contract['manifest_hash']
    assert config['serialization'] == 'modernbert-oracle-v2-cls-sep'
    assert config['backbone'] == 'frozen_eval_fp32_sdpa_reference_compile_false'
    assert json.loads((path / 'run-resources/exit.json').read_text())['exit_code'] == 0
    groups = {}
    for row in rows:
        if groups.setdefault(row['group_id'], row['split']) != row['split']:
            raise ValueError('来源组跨split')
    features, hashes = {}, {}
    for view in ['G', 'E']:
        meta = json.loads((path / f'{view}_cache.json').read_text())
        x = np.load(path / f'{view}_features.npy', mmap_mode='r')
        done = np.load(path / f'{view}_done.npy', mmap_mode='r')
        active = np.array([view == 'G' or row['attack'] for row in rows])
        assert meta['contract_hash'] == digest(contract) and list(x.shape) == meta['shape']
        assert len(x) == len(rows) and done[active].all() and np.isfinite(x[active]).all()
        features[view] = x
        hashes[view] = file_sha256(path / f'{view}_features.npy')
    identity = {'path': str(path.resolve()), 'model': config['model'], 'hidden_size': features['G'].shape[1],
                'manifest_sha256': file_sha256(path / 'manifest.jsonl'), 'contract_sha256': digest(contract),
                'source_data': contract['data'], 'feature_sha256': hashes}
    return rows, features, identity


def replay_scores(head_dir, features, rows, name):
    checkpoint = torch.load(Path(head_dir) / 'best.pt', map_location='cpu', weights_only=True)
    assert checkpoint['label_mapping'] == {'A': 1, 'nonA': 0}
    head = torch.nn.Linear(checkpoint['hidden_size'], 1)
    head.load_state_dict(checkpoint['state_dict'])
    head.eval()
    valid = np.array([i for i, row in enumerate(rows) if name != 'E' or row['attack']])
    output = np.full(len(rows), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for offset in range(0, len(valid), 256):
            idx = valid[offset:offset + 256]
            output[idx] = head(torch.from_numpy(np.asarray(features[idx]).copy())).squeeze(-1).sigmoid().numpy()
    return output


def capacity_bootstrap(y, predictions, groups, seed, iterations=500):
    unique = sorted(set(groups))
    matrices = {name: np.stack([np.bincount(y[groups == group] * 2 + pred[groups == group], minlength=4).reshape(2, 2)
                               for group in unique]) for name, pred in predictions.items()}
    pairs = [(f'large_{m}', f'base_{m}') for m in ['M0', 'M1', 'M2']]
    pairs += [(f'{model}_M2', f'{model}_{m}') for model in ['base', 'large'] for m in ['M0', 'M1']]
    samples = {f'{a}-minus-{b}/{metric}': [] for a, b in pairs for metric in ['A_recall', 'A_fpr']}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        select = rng.integers(0, len(unique), len(unique))
        stats = {}
        for name, cm in matrices.items():
            tn, fp, fn, tp = cm[select].sum(axis=0).ravel()
            stats[name] = {'A_recall': tp / (tp + fn) if tp + fn else np.nan,
                           'A_fpr': fp / (fp + tn) if fp + tn else np.nan}
        for a, b in pairs:
            for metric in ['A_recall', 'A_fpr']:
                samples[f'{a}-minus-{b}/{metric}'].append(stats[a][metric] - stats[b][metric])
    return {'groups': len(unique), 'unit': 'source_group', 'iterations': iterations,
            'percentile_95': {k: np.nanquantile(v, [.025, .975]).tolist() for k, v in samples.items()}}


def evaluate(out, rows, feature_sets, config):
    y = binary_labels([r['y'] for r in rows])
    enabled = np.array([r['attack'] for r in rows])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test'])
    original = np.array([rows[i]['y'] for i in test])
    results = {'seeds': {}, 'bootstrap': {}, 'selection': config['checkpoint_selection']}
    with (out / 'test_predictions.jsonl').open('w') as f:
        for seed in config['seeds']:
            all_predictions = {}
            results['seeds'][str(seed)] = {}
            for model in ['base', 'large']:
                probabilities = {name: replay_scores(out / model / f'{name}-seed-{seed}',
                                 feature_sets[model]['E' if name == 'E' else 'G'], rows, name)
                                 for name in ['G', 'G_aux', 'E']}
                scores = {'M0': probabilities['G'],
                          'M1': aggregate(probabilities['G'], probabilities['G_aux'], enabled),
                          'M2': aggregate(probabilities['G'], probabilities['E'], enabled)}
                entries = {}
                for name, score in scores.items():
                    threshold = fast_calibrate(y[val], score[val], config['max_a_fpr'])
                    pred = predict_binary(score[test], threshold)
                    all_predictions[model + '_' + name] = pred
                    strata = {}
                    for field in ['domain', 'variant', 'label_source', 'group_id']:
                        group = np.array([rows[i][field] for i in test])
                        strata[field] = {str(g): binary_metrics(y[test][group == g], pred[group == g]) for g in sorted(set(group))}
                    entries[name] = {'threshold_A': threshold, 'validation': binary_metrics(y[val], predict_binary(score[val], threshold)),
                                     'test': binary_metrics(y[test], pred), 'strata': strata,
                                     'B_to_A_fpr': float(pred[original == 1].mean()),
                                     'C_to_A_fpr': float(pred[original == 2].mean())}
                    for j, index in enumerate(test):
                        f.write(json.dumps({'model': model, 'seed': seed, 'configuration': name, 'id': rows[index]['id'],
                                            'group_id': rows[index]['group_id'], 'y_A': int(y[index]),
                                            'score_A': float(score[index]), 'prediction_A': int(pred[j])}) + '\n')
                results['seeds'][str(seed)][model] = entries
            results['bootstrap'][str(seed)] = capacity_bootstrap(y[test], all_predictions,
                np.array([rows[i]['group_id'] for i in test]), seed)
            f.flush()
    results['means'] = {model: {name: {metric: statistics.mean(r[model][name]['test'][metric] for r in results['seeds'].values())
                                     for metric in ['A_recall', 'A_fpr', 'A_precision', 'A_f1']}
                               for name in ['M0', 'M1', 'M2']} for model in ['base', 'large']}
    (out / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def report(out, results, summaries, elapsed):
    lines = ['# ModernBERT-large 对 base：冻结特征二分类＋完整训练轨迹', '',
             '两种骨干使用相同数据、相同token输入与划分；每个分类头都跑满100epoch。只根据未加权validation BCE选择最佳checkpoint。',
             '这是已看过测试集上的探索性oracle证据实验；不进行逐epoch测试，不根据测试结果选择训练长度。', '',
             '## 三种子测试均值', '',
             '| 方案 | base A召回 | large A召回 | base A误报 | large A误报 | base A精确率 | large A精确率 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in ['M0', 'M1', 'M2']:
        b, l = results['means']['base'][name], results['means']['large'][name]
        lines.append(f'| {name} | {b["A_recall"]:.2%} | {l["A_recall"]:.2%} | {b["A_fpr"]:.2%} | {l["A_fpr"]:.2%} | {b["A_precision"]:.2%} | {l["A_precision"]:.2%} |')
    lines += ['', '## 完整训练轨迹', '',
              '| 骨干 | 头 | seed | 最佳epoch | patience10模拟停止epoch | 最佳val BCE | 第100轮val BCE | 末轮高于最佳 |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for model in ['base', 'large']:
        for s in summaries[model]:
            lines.append(f'| {model} | {s["name"]} | {s["seed"]} | {s["best_epoch"]} | {s["simulated_patience10_stop_epoch"]} | '
                         f'{s["best_val_bce"]:.4f} | {s["final_val_bce"]:.4f} | {s["final_val_excess_percent"]:.1f}% |')
    late = sum(s['best_in_last20'] for ss in summaries.values() for s in ss)
    lines += ['', f'- 18个头都完成100轮；其中{late}个头的最佳验证BCE出现在最后20轮。',
              '- 不把“训练损失仍下降”直接解释成“继续训练能提升测试效果”；查看train/val损失是否分离、验证最佳轮次及最后20轮斜率。',
              '- best.pt由全100轮验证BCE最小值选择，final.pt是第100轮；二者不是同一个概念。',
              '- 每epoch保存训练与验证未加权/加权BCE、梯度范数、学习率、验证5%FPR工作点及验证预测。',
              '- 每epoch保存权重、优化器状态、PyTorch/shuffle RNG；包括初始epoch000，共每头101份checkpoint，另有best/final。',
              '- 路径：`{base,large}/{G,G_aux,E}-seed-{42,43,44}/trajectory.{csv,jsonl,svg}`；验证预测为`validation_scores.npy`，列对应`validation_manifest.jsonl`。',
              '- `summary.json`含最后20轮斜率、初始/最佳/最终损失、模拟早停轮次；`checkpoints/epoch-XXX.pt`可追溯任一轮。',
              '- 全程训练不早停，只记录patience10的模拟停止位置，避免混淆模型大小和实际训练长度。', '',
              '## 实验设置与边界', '',
              '- 共享输入定义、固定来源组划分，未新增样本。两种骨干token哈希逐行一致；只有编码器及输出表示维度变化。',
              '- 冻结FP32骨干、末层非padding mean pooling。只训练线性二分类头，不是微调整个ModernBERT。',
              '- M0=G；M1=G+G_aux；M2=G+E。附加分支仅在已知攻击原文非空时启用。',
              '- AdamW lr1e-3、weight_decay1e-4、batch128；BCE pos_weight由各头train负/正比计算，validation使用自然分布。',
              '- 每方案在验证集A误报≤5%时最大化召回；测试误报率不能人为调到5%。分支max风险分数不宣称已校准概率。',
              '- 旧base二分类启用早停，与这轮base全100轮不是完全相同的训练规则；正式比较以本轮base/large为准。',
              '- test此前已使用，后续确认泛化需新的留出集；同一来源多个块相关，配对bootstrap按来源组进行。',
              f'- 本轮CPU轨迹训练与评估逻辑耗时{elapsed:.2f}秒；large特征提取资源及耗时另见特征源日志。', '']
    (out / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', default='artifacts/modernbert_oracle_poc/run-20261104-v2')
    parser.add_argument('--large-run', default='artifacts/modernbert_oracle_poc/large-features-20261105-v2')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('分类头训练需CPU-only受限wrapper')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        raise ValueError('不得覆盖既有运行，使用新目录')
    rows, base, base_identity = load_features(args.base_run)
    large_rows, large, large_identity = load_features(args.large_run)
    if rows != large_rows or base_identity['source_data'] != large_identity['source_data']:
        raise ValueError('base/large样本或实际token输入不同，停止而非静默比较')
    assert base_identity['model']['model'] == 'answerdotai/ModernBERT-base'
    assert large_identity['model']['model'] == 'answerdotai/ModernBERT-large'
    feature_sets = {'base': base, 'large': large}
    identities = {'base': base_identity, 'large': large_identity}
    config = {'training': TRAIN_CONFIG, 'features': identities, 'manifest_hash': digest(rows),
              'protocol_sha256': file_sha256(out / 'protocol.json'),
              'code_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in
                              ['modernbert_capacity.py', 'modernbert_trajectory.py', 'modernbert_binary.py', 'modernbert_data.py']}}
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    summaries = {'base': [], 'large': []}
    for model in ['base', 'large']:
        for seed in TRAIN_CONFIG['seeds']:
            for name in ['G', 'G_aux', 'E']:
                summary = train_trajectory(name, feature_sets[model]['E' if name == 'E' else 'G'], rows, seed,
                                           out / model / f'{name}-seed-{seed}', TRAIN_CONFIG, identities[model])
                summaries[model].append(summary)
    (out / 'trajectory_summary.json').write_text(json.dumps(summaries, indent=2))
    results = evaluate(out, rows, feature_sets, TRAIN_CONFIG)
    elapsed = time.perf_counter() - started
    report(out, results, summaries, elapsed)
    (out / 'completed.json').write_text(json.dumps({'config_hash': digest(config), 'elapsed_seconds': elapsed}))
    print(f'base/large完整轨迹对照已完成：{out / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
