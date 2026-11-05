"""冻结特征上的线性头、仅验证集校准、来源组配对 bootstrap。"""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


def confusion(y, pred):
    return np.bincount(np.asarray(y) * 3 + np.asarray(pred), minlength=9).reshape(3, 3)


def metrics(y, pred):
    cm = confusion(y, pred)
    support = cm.sum(axis=1)
    recalls = np.divide(cm.diagonal(), support, out=np.zeros(3, dtype=float), where=support > 0)
    precision = np.divide(cm.diagonal(), cm.sum(axis=0), out=np.zeros(3, dtype=float), where=cm.sum(axis=0) > 0)
    f1 = np.divide(2 * recalls * precision, recalls + precision,
                   out=np.zeros(3, dtype=float), where=(recalls + precision) > 0)
    neg = int(support[1:].sum())
    return {'n': int(len(y)), 'confusion': cm.tolist(), 'A_recall': float(recalls[0]) if support[0] else None,
            'A_fpr': float(cm[1:, 0].sum() / neg) if neg else None,
            'B_recall': float(recalls[1]) if support[1] else None,
            'B_f1': float(f1[1]), 'macro_f1': float(f1.mean()),
            'C_to_A_or_B': float(cm[2, :2].sum() / support[2]) if support[2] else None}


def combined_scores(g, aux, enabled):
    scores = np.asarray(g).copy()
    if aux is not None:
        scores[enabled] = np.maximum(scores[enabled], aux[enabled])
    return scores


def predict(scores, tau_a, tau_b):
    scores = np.asarray(scores, dtype=np.float64)
    return np.where(scores[:, 0] >= tau_a, 0,
                    np.where(scores[:, 1] >= tau_b, 1, 2))


def calibrate(y, scores, max_fpr=0.05):
    """只接收val；先A召回/FPR/阈值，再最终B-F1/阈值，确定性处理平手。"""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.any(y == 0) or not np.any(y != 0):
        raise ValueError('校准集缺少A或A负例')
    thresholds_a = np.r_[np.nextafter(float(scores[:, 0].max()), np.inf), np.unique(scores[:, 0])]
    best = None
    for threshold in thresholds_a:
        positive = scores[:, 0] >= threshold
        fpr = float(positive[y != 0].mean())
        recall = float(positive[y == 0].mean())
        key = (recall, -fpr, float(threshold))
        if fpr <= max_fpr + 1e-12 and (best is None or key > best[0]):
            best = (key, float(threshold))
    tau_a = best[1]
    best_b = None
    for threshold in np.r_[np.nextafter(float(scores[:, 1].max()), np.inf), np.unique(scores[:, 1])]:
        pred = predict(scores, tau_a, threshold)
        key = (metrics(y, pred)['B_f1'], float(threshold))
        if best_b is None or key > best_b[0]:
            best_b = (key, float(threshold))
    return tau_a, best_b[1]


def train_head(name, features, rows, seed, out_dir, config):
    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train' and (name == 'G' or r['attack'])])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val' and (name == 'G' or r['attack'])])
    y = np.array([r['y'] for r in rows])
    x_train = torch.from_numpy(np.asarray(features[train]).copy())
    x_val = torch.from_numpy(np.asarray(features[val]).copy())
    y_train, y_val = torch.tensor(y[train]), torch.tensor(y[val])
    counts = torch.bincount(y_train, minlength=3)
    if (counts == 0).any():
        raise ValueError('训练子集有空类别')
    weight = len(train) / (3 * counts.float())
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = nn.Linear(x_train.shape[1], 3)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    best_loss, best_state, best_epoch, stale = float('inf'), None, 0, 0
    curve = []
    start = time.perf_counter()
    for epoch in range(1, config['max_epochs'] + 1):
        head.train()
        indices = torch.randperm(len(train), generator=generator)
        loss_sum = 0.0
        for batch in indices.split(config['head_batch']):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        head.eval()
        with torch.inference_mode():
            val_loss = float(nn.functional.cross_entropy(head(x_val), y_val))
        curve.append({'epoch': epoch, 'train_weighted_ce': loss_sum / len(train), 'val_ce': val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= config['patience']:
            break
    head.load_state_dict(best_state)
    output = np.full((len(rows), 3), np.nan, dtype=np.float32)
    valid = np.array([i for i, row in enumerate(rows) if name != 'E' or row['attack']])
    with torch.inference_mode():
        for offset in range(0, len(valid), 256):
            idx = valid[offset:offset + 256]
            output[idx] = head(torch.from_numpy(np.asarray(features[idx]).copy())).softmax(-1).numpy()
    checkpoint = Path(out_dir) / f'{name}-seed-{seed}.pt'
    torch.save({'state_dict': best_state, 'hidden_size': x_train.shape[1], 'classes': ['A', 'B', 'C'],
                'seed': seed, 'name': name, 'train_class_counts': counts.tolist(),
                'train_class_weights': weight.tolist(), 'best_epoch': best_epoch,
                'config': config}, checkpoint)
    info = {'name': name, 'seed': seed, 'train_n': len(train), 'val_n': len(val),
            'train_class_counts': counts.tolist(), 'class_weights': weight.tolist(),
            'best_epoch': best_epoch, 'best_val_ce': best_loss,
            'elapsed_seconds': time.perf_counter() - start, 'curve': curve}
    checkpoint.with_suffix('.json').write_text(json.dumps(info, indent=2))
    print(f'训练完成 {name} seed={seed} epoch={best_epoch} seconds={info["elapsed_seconds"]:.1f}', flush=True)
    return output, info


def group_bootstrap(y, predictions, groups, seed, iterations=500):
    """对来源组有放回抽样，三配置配对；区间不把相邻块视为独立。"""
    unique = sorted(set(groups))
    indices = [np.flatnonzero(groups == g) for g in unique]
    cms = {name: np.stack([confusion(y[ix], pred[ix]) for ix in indices])
           for name, pred in predictions.items()}
    rng = np.random.default_rng(seed)
    samples = {name + '/' + metric: [] for name in predictions for metric in ['A_recall', 'A_fpr']}
    for baseline in ['M0', 'M1']:
        for metric in ['A_recall', 'A_fpr']:
            samples['M2-minus-' + baseline + '/' + metric] = []
    for _ in range(iterations):
        selection = rng.integers(0, len(unique), len(unique))
        stats = {}
        for name, matrices in cms.items():
            cm = matrices[selection].sum(axis=0)
            denom_a, denom_neg = cm[0].sum(), cm[1:].sum()
            vals = {'A_recall': cm[0, 0] / denom_a if denom_a else float('nan'),
                    'A_fpr': cm[1:, 0].sum() / denom_neg if denom_neg else float('nan')}
            stats[name] = vals
            for metric, value in vals.items():
                samples[name + '/' + metric].append(value)
        for baseline in ['M0', 'M1']:
            for metric in ['A_recall', 'A_fpr']:
                samples['M2-minus-' + baseline + '/' + metric].append(stats['M2'][metric] - stats[baseline][metric])
    return {'unit': 'source_group', 'groups': len(unique), 'iterations': iterations,
            'percentile_95': {key: np.nanquantile(values, [0.025, 0.975]).tolist() for key, values in samples.items()}}


def run_heads(features_g, features_e, rows, out_dir, config):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    y = np.array([r['y'] for r in rows])
    enabled = np.array([r['attack'] for r in rows])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test'])
    results = {'config': config, 'seeds': {}, 'limitations': [
        '离线块级指标，不等于在线防御率。', 'oracle攻击证据与路由，不可直接部署。',
        'validation复用于早停和阈值校准，小样本有选择偏差。',
        'M1复用G编码；M2需要E额外编码，不是等推理成本。',
        '仅检验冻结表示的线性可读性，不代表骨干微调的能力。']}
    with (out_dir / 'test_predictions.jsonl').open('w') as f:
        for seed in config['seeds']:
            g, _ = train_head('G', features_g, rows, seed, out_dir, config)
            aux, _ = train_head('G_aux', features_g, rows, seed, out_dir, config)
            e, _ = train_head('E', features_e, rows, seed, out_dir, config)
            scores = {'M0': g, 'M1': combined_scores(g, aux, enabled),
                      'M2': combined_scores(g, e, enabled)}
            seed_result, predictions = {}, {}
            for name, score in scores.items():
                tau_a, tau_b = calibrate(y[val], score[val], config['max_a_fpr'])
                pred = predict(score[test], tau_a, tau_b)
                predictions[name] = pred
                strata = {}
                for field in ['domain', 'variant', 'label_source', 'group_id']:
                    groups = np.array([rows[i][field] for i in test])
                    strata[field] = {value: metrics(y[test][groups == value], pred[groups == value])
                                     for value in sorted(set(groups))}
                seed_result[name] = {'tau_A': tau_a, 'tau_B': tau_b,
                                     'validation': metrics(y[val], predict(score[val], tau_a, tau_b)),
                                     'test': metrics(y[test], pred), 'strata': strata}
                for j, index in enumerate(test):
                    f.write(json.dumps({'seed': seed, 'configuration': name, **rows[index],
                                        'score_A': float(score[index, 0]), 'score_B': float(score[index, 1]),
                                        'prediction': int(pred[j])}) + '\n')
            yt = y[test]
            for base in ['M0', 'M1']:
                new = (predictions['M2'] == 0) & (predictions[base] != 0)
                lost = (predictions['M2'] != 0) & (predictions[base] == 0)
                seed_result['M2-vs-' + base] = {
                    'new_A_true_positives': int((new & (yt == 0)).sum()),
                    'new_A_false_positives': int((new & (yt != 0)).sum()),
                    'lost_A_true_positives': int((lost & (yt == 0)).sum()),
                    'removed_A_false_positives': int((lost & (yt != 0)).sum())}
            seed_result['bootstrap'] = group_bootstrap(yt, predictions,
                np.array([rows[i]['group_id'] for i in test]), seed, config['bootstrap_iterations'])
            results['seeds'][str(seed)] = seed_result
            (out_dir / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
            f.flush()
    return results
