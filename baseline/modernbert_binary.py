"""复用已核验v2特征的 A/非A 对照；不重新编码、不扩充样本、不修改原三分类结果。"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch import nn

from baseline.modernbert_data import digest, file_sha256, json_lines

CONFIG = {'lr': 1e-3, 'weight_decay': 1e-4, 'head_batch': 128, 'max_epochs': 100,
          'patience': 10, 'seeds': [42, 43, 44], 'max_a_fpr': .05,
          'bootstrap_iterations': 500, 'output': 'one_logit_sigmoid_A',
          'loss': 'BCEWithLogitsLoss(pos_weight=train_nonA/train_A)',
          'early_stop': 'unweighted_validation_BCE',
          'sample_scope': 'identical_to_verified_three_class_v2',
          'test_status': '已看过的同一测试集，仅作探索性对照，不是新的盲测'}


def binary_labels(original):
    original = np.asarray(original)
    if not np.isin(original, [0, 1, 2]).all():
        raise ValueError('二分类第一阶段不接收未知原标签')
    return (original == 0).astype(np.int64)  # 1=A，0=非A；不同于原class_id=0表示A。


def predict_binary(scores, threshold):
    return (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int64)


def binary_metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError('需要二分类0/1编码')
    cm = np.bincount(y * 2 + pred, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = map(int, cm.ravel())
    return {'n': len(y), 'support_A': tp + fn, 'support_nonA': tn + fp,
            'confusion_order': ['nonA', 'A'], 'confusion': cm.tolist(),
            'A_recall': tp / (tp + fn) if tp + fn else None,
            'A_fpr': fp / (fp + tn) if fp + tn else None,
            'A_precision': tp / (tp + fp) if tp + fp else 0.0,
            'A_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def calibrate_binary(y, scores, max_fpr=.05):
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all() or not np.any(y == 1) or not np.any(y == 0):
        raise ValueError('校准分数不合法，或缺少正/负类')
    best = None
    for threshold in np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)]:
        pred = scores >= threshold
        recall, fpr = float(pred[y == 1].mean()), float(pred[y == 0].mean())
        key = (recall, -fpr, float(threshold))
        if fpr <= max_fpr + 1e-12 and (best is None or key > best[0]):
            best = (key, float(threshold))
    return best[1]


def aggregate(g, aux, enabled):
    score = np.asarray(g).copy()
    if aux is not None:
        score[enabled] = np.maximum(score[enabled], aux[enabled])
    return score


def train_binary_head(name, features, rows, seed, out_dir, config):
    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train' and (name == 'G' or r['attack'])])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val' and (name == 'G' or r['attack'])])
    y = binary_labels([r['y'] for r in rows])
    xt = torch.from_numpy(np.asarray(features[train]).copy())
    xv = torch.from_numpy(np.asarray(features[val]).copy())
    yt, yv = torch.tensor(y[train], dtype=torch.float32), torch.tensor(y[val], dtype=torch.float32)
    positives = int(yt.sum())
    negatives = len(train) - positives
    if not positives or not negatives:
        raise ValueError('训练子集缺少A或非A')
    pos_weight = negatives / positives
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = nn.Linear(xt.shape[1], 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    best_loss, best_state, stale, best_epoch = float('inf'), None, 0, 0
    curve = []
    start = time.perf_counter()
    for epoch in range(1, config['max_epochs'] + 1):
        total = 0.0
        head.train()
        for batch in torch.randperm(len(train), generator=generator).split(config['head_batch']):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(xt[batch]).squeeze(-1), yt[batch])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        head.eval()
        with torch.inference_mode():
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(head(xv).squeeze(-1), yv))
        curve.append({'epoch': epoch, 'train_weighted_bce': total / len(train), 'val_unweighted_bce': val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= config['patience']:
            break
    head.load_state_dict(best_state)
    scores = np.full(len(rows), np.nan, dtype=np.float32)
    valid = np.array([i for i, r in enumerate(rows) if name != 'E' or r['attack']])
    with torch.inference_mode():
        for offset in range(0, len(valid), 256):
            idx = valid[offset:offset + 256]
            scores[idx] = head(torch.from_numpy(np.asarray(features[idx]).copy())).squeeze(-1).sigmoid().numpy()
    out_dir = Path(out_dir)
    torch.save({'state_dict': best_state, 'hidden_size': xt.shape[1], 'output_size': 1,
                'label_mapping': {'nonA': 0, 'A': 1}, 'output': 'sigmoid(logit)=score_A',
                'seed': seed, 'name': name, 'train_counts': {'A': positives, 'nonA': negatives},
                'pos_weight': pos_weight, 'best_epoch': best_epoch, 'config': config},
               out_dir / f'{name}-seed-{seed}.pt')
    info = {'name': name, 'seed': seed, 'train_n': len(train), 'val_n': len(val),
            'train_counts': {'A': positives, 'nonA': negatives}, 'pos_weight': pos_weight,
            'best_epoch': best_epoch, 'best_val_bce': best_loss, 'curve': curve,
            'elapsed_seconds': time.perf_counter() - start}
    (out_dir / f'{name}-seed-{seed}.json').write_text(json.dumps(info, indent=2))
    print(f'二分类 {name} seed={seed} best_epoch={best_epoch} seconds={info["elapsed_seconds"]:.3f}', flush=True)
    return scores, info


def paired_bootstrap(y, predictions, groups, seed, iterations):
    names = list(predictions)
    unique = sorted(set(groups))
    matrices = {}
    for name, pred in predictions.items():
        matrices[name] = np.stack([np.bincount(y[groups == g] * 2 + pred[groups == g], minlength=4).reshape(2, 2)
                                   for g in unique])
    pairs = [('M2', 'M0'), ('M2', 'M1')] + [(m, 'three_' + m) for m in ['M0', 'M1', 'M2']]
    pairs = [(a, b) for a, b in pairs if a in predictions and b in predictions]
    samples = {f'{a}-minus-{b}/{metric}': [] for a, b in pairs for metric in ['A_recall', 'A_fpr']}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        select = rng.integers(0, len(unique), len(unique))
        values = {}
        for name in names:
            tn, fp, fn, tp = matrices[name][select].sum(axis=0).ravel()
            values[name] = {'A_recall': tp / (tp + fn) if tp + fn else np.nan,
                            'A_fpr': fp / (fp + tn) if fp + tn else np.nan}
        for a, b in pairs:
            for metric in ['A_recall', 'A_fpr']:
                samples[f'{a}-minus-{b}/{metric}'].append(values[a][metric] - values[b][metric])
    return {'unit': 'source_group', 'groups': len(unique), 'iterations': iterations,
            'percentile_95': {key: np.nanquantile(value, [.025, .975]).tolist() for key, value in samples.items()}}


def load_source(source):
    source = Path(source)
    if (source / 'INVALID.json').exists():
        raise ValueError('不能复用无效调试轮')
    for filename in ['completed.json', 'verification.json', 'config.json', 'cache_contract.json']:
        if not (source / filename).exists():
            raise ValueError('原三分类实验未完成/未核验')
    cfg = json.loads((source / 'config.json').read_text())
    completed = json.loads((source / 'completed.json').read_text())
    assert completed['config_hash'] == digest(cfg)
    assert cfg['serialization'] == 'modernbert-oracle-v2-cls-sep'
    rows = list(json_lines(source / 'manifest.jsonl'))
    contract = json.loads((source / 'cache_contract.json').read_text())
    assert digest(rows) == contract['manifest_hash']
    assert contract['config'] == cfg
    groups = {}
    for row in rows:
        if groups.setdefault(row['group_id'], row['split']) != row['split']:
            raise ValueError('来源组跨split')
    features = {}
    for view in ['G', 'E']:
        metadata = json.loads((source / f'{view}_cache.json').read_text())
        assert metadata['contract_hash'] == digest(contract)
        x = np.load(source / f'{view}_features.npy', mmap_mode='r')
        done = np.load(source / f'{view}_done.npy', mmap_mode='r')
        eligible = np.array([view == 'G' or row['attack'] for row in rows])
        assert list(x.shape) == metadata['shape'] and len(x) == len(rows)
        assert done[eligible].all() and np.isfinite(x[eligible]).all()
        features[view] = x
    return rows, features, cfg


def run_experiment(source, out, config=CONFIG):
    start = time.perf_counter()
    rows, features, source_config = load_source(source)
    y = binary_labels([r['y'] for r in rows])
    enabled = np.array([r['attack'] for r in rows])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test'])
    original = np.array([rows[i]['y'] for i in test])
    artifact = Path(out)
    artifact.mkdir(parents=True, exist_ok=True)
    if (artifact / 'config.json').exists():
        raise ValueError('使用新的二分类输出目录，不覆盖任何既有实验')
    source = Path(source)
    source_files = ['config.json', 'manifest.jsonl', 'cache_contract.json', 'G_features.npy', 'E_features.npy',
                    'heads/results.json', 'heads/test_predictions.jsonl']
    provenance = {'source_run': str(source.resolve()), 'source_files_sha256': {f: file_sha256(source / f) for f in source_files},
                  'source_config': source_config, 'head': config,
                  'code_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in
                                  ['modernbert_binary.py', 'modernbert_data.py']},
                  'label_mapping': {'A': 1, 'nonA': 0},
                  'counts': {split: {'A': sum(r['split'] == split and r['y'] == 0 for r in rows),
                                      'nonA': sum(r['split'] == split and r['y'] != 0 for r in rows)}
                             for split in ['train', 'val', 'test']}}
    (artifact / 'config.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    heads = artifact / 'heads'
    heads.mkdir(exist_ok=True)
    previous = {(r['seed'], r['configuration'], r['id']): r for r in json_lines(source / 'heads/test_predictions.jsonl')}
    previous_results = json.loads((source / 'heads/results.json').read_text())
    result = {'config': config, 'seeds': {}, 'head_training_seconds': 0.0}
    with (heads / 'test_predictions.jsonl').open('w') as f:
        for seed in config['seeds']:
            probabilities = {}
            for name in ['G', 'G_aux', 'E']:
                scores, info = train_binary_head(name, features['E' if name == 'E' else 'G'], rows, seed, heads, config)
                probabilities[name] = scores
                result['head_training_seconds'] += info['elapsed_seconds']
            scores = {'M0': probabilities['G'],
                      'M1': aggregate(probabilities['G'], probabilities['G_aux'], enabled),
                      'M2': aggregate(probabilities['G'], probabilities['E'], enabled)}
            entries, predictions = {}, {}
            for name, score in scores.items():
                threshold = calibrate_binary(y[val], score[val], config['max_a_fpr'])
                pred = predict_binary(score[test], threshold)
                predictions[name] = pred
                old_rows = [previous[(seed, name, rows[i]['id'])] for i in test]
                assert [r['y'] for r in old_rows] == original.tolist()
                old = np.array([int(r['prediction'] == 0) for r in old_rows])
                predictions['three_' + name] = old
                strata = {}
                for field in ['domain', 'variant', 'label_source', 'group_id']:
                    groups = np.array([rows[i][field] for i in test])
                    strata[field] = {str(g): binary_metrics(y[test][groups == g], pred[groups == g]) for g in sorted(set(groups))}
                new = (pred == 1) & (old == 0)
                lost = (pred == 0) & (old == 1)
                entries[name] = {'threshold': threshold, 'validation': binary_metrics(y[val], predict_binary(score[val], threshold)),
                                 'test': binary_metrics(y[test], pred), 'three_class_test': binary_metrics(y[test], old),
                                 'three_class_threshold_A': previous_results['seeds'][str(seed)][name]['tau_A'],
                                 'B_to_A_fpr': float(pred[original == 1].mean()) if np.any(original == 1) else None,
                                 'C_to_A_fpr': float(pred[original == 2].mean()) if np.any(original == 2) else None,
                                 'versus_three': {'new_true_A': int((new & (y[test] == 1)).sum()),
                                                  'new_false_A': int((new & (y[test] == 0)).sum()),
                                                  'lost_true_A': int((lost & (y[test] == 1)).sum()),
                                                  'removed_false_A': int((lost & (y[test] == 0)).sum())},
                                 'strata': strata}
                for j, i in enumerate(test):
                    f.write(json.dumps({'seed': seed, 'configuration': name, 'id': rows[i]['id'],
                                        'group_id': rows[i]['group_id'], 'original_class_id': rows[i]['y'],
                                        'y_A': int(y[i]), 'prediction_A': int(pred[j]),
                                        'score_A': float(score[i]), 'three_prediction_A': int(old[j])}) + '\n')
            entries['bootstrap'] = paired_bootstrap(y[test], predictions, np.array([rows[i]['group_id'] for i in test]),
                                                     seed, config['bootstrap_iterations'])
            result['seeds'][str(seed)] = entries
            f.flush()
            (heads / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    result['elapsed_seconds'] = time.perf_counter() - start
    (heads / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    report = ['# A／非A 二分类探索性对照', '',
              '相同v2特征、相同样本和划分，只改变标签/输出/损失；没有重新编码、扩充数据或使用GPU。',
              'A编码为1，非A编码为0；原三分类class_id=0表示A，代码显式转换，不能混用。', '',
              '## 三种子均值', '',
              '| 配置 | 三分类A召回 | 二分类A召回 | 三分类A误报 | 二分类A误报 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name in ['M0', 'M1', 'M2']:
        mean = lambda key, metric: statistics.mean(v[name][key][metric] for v in result['seeds'].values())
        report.append(f'| {name} | {mean("three_class_test", "A_recall"):.2%} | {mean("test", "A_recall"):.2%} | '
                      f'{mean("three_class_test", "A_fpr"):.2%} | {mean("test", "A_fpr"):.2%} |')
    report += ['', '## 各种子', '', '| seed | 配置 | A召回 | A误报率 | A精确率 | 验证阈值 |',
               '| --- | --- | ---: | ---: | ---: | ---: |']
    for seed, entries in result['seeds'].items():
        for name in ['M0', 'M1', 'M2']:
            x = entries[name]
            report.append(f'| {seed} | {name} | {x["test"]["A_recall"]:.2%} | {x["test"]["A_fpr"]:.2%} | '
                          f'{x["test"]["A_precision"]:.2%} | {x["threshold"]:.6f} |')
    report += ['', '## 固定设置与局限', '',
               '- M0=通用头；M1=通用头＋攻击子集通用头；M2=通用头＋攻击证据头。仅attack_str非空时启用附加头。',
               '- 骨干冻结、单输出Linear；BCEWithLogitsLoss的pos_weight仅取训练负/正样本比；AdamW、lr=1e-3、weight_decay=1e-4。',
               '- CPU batch128，最多100epoch、patience10；按各自验证范围的未加权BCE选择checkpoint；seeds42/43/44串行。',
               '- 相比三分类，不再分别平衡B/C，损失及类别权重同时随二分类定义改变；不能把收益完全归因于参数量减少。',
               '- 分支取max(score_A)，每配置独立在验证集A误报≤5%时最大化召回；平手优先低误报/高阈值。',
               '- 加权损失的sigmoid与分支max仅当作风险分数，不声称概率已校准。实际测试FPR可以超过5%。',
               '- validation复用于早停和阈值选择；同一test此前已看过，本轮是探索性比较，最终结论需要新留出集。',
               '- 来源组配对bootstrap区间见heads/results.json；三种子不是三个独立测试集，不按test成绩挑种子。',
               '- oracle攻击证据，不是可部署的攻击定位器；非A仅代表当前主模型未中招，不代表没有攻击或全任务安全。',
               '- 本轮未利用原三分类过滤掉的“草稿未知但主模型标签明确”样本。',
               f'- 九个小头训练及头预测共{result["head_training_seconds"]:.2f}秒；实验逻辑共{result["elapsed_seconds"]:.2f}秒，不含Python启动。',
               '- 检查点位于heads/{G,G_aux,E}-seed-{42,43,44}.pt；使用config.json指定的原v2骨干/输入协议与本轮阈值。', '']
    (artifact / 'report.md').write_text('\n'.join(report))
    (artifact / 'completed.json').write_text(json.dumps({'config_hash': digest(provenance), 'elapsed_seconds': result['elapsed_seconds']}))
    print(f'二分类实验完成：{artifact / "report.md"}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', default='artifacts/modernbert_oracle_poc/run-20261104-v2')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('需要受限wrapper且CUDA_VISIBLE_DEVICES=空，保证仅CPU执行')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    run_experiment(args.source_run, args.run_dir)


if __name__ == '__main__':
    main()
