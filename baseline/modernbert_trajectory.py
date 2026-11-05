"""冻结特征二分类头的完整训练轨迹；只用validation选checkpoint，test不进入此模块。"""
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from baseline.modernbert_binary import binary_labels, binary_metrics


def fast_calibrate(y, scores, max_fpr=.05):
    """与逐阈值穷举完全相同的排序/平手规则，O(n log n)。"""
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=np.float64)
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all() or not 0 < y.sum() < len(y):
        raise ValueError('需要有限分数和非空正负类')
    order = np.argsort(-scores, kind='stable')
    sorted_scores, sorted_y = scores[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(y) - 1]
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), sorted_scores[ends]]
    recall = np.r_[0., np.cumsum(sorted_y)[ends] / y.sum()]
    fpr = np.r_[0., np.cumsum(1 - sorted_y)[ends] / (len(y) - y.sum())]
    valid = np.flatnonzero(fpr <= max_fpr + 1e-12)
    best = valid[np.lexsort((thresholds[valid], -fpr[valid], recall[valid]))[-1]]
    return float(thresholds[best])


def atomic_torch_save(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def trajectory_svg(curve, best_epoch, path):
    """无需额外绘图库：保存可直接打开的训练/验证损失和验证工作点曲线。"""
    width, height = 820, 540
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<style>text{font-family:sans-serif;font-size:12px}</style>']
    panels = [(40, 220, [('train_bce', '#1768ac'), ('val_bce', '#d1495b')], 'Unweighted BCE (train / validation)'),
              (300, 200, [('val_A_recall', '#16845b'), ('val_A_fpr', '#a65c00')], 'Validation recall / FPR at FPR <= 5%')]
    max_epoch = max(r['epoch'] for r in curve)
    for top, panel_height, series, title in panels:
        left, plot_width = 65, 720
        high = max(max(r[key] for r in curve) for key, _ in series) * 1.05
        high = max(high, .1)
        elements.append(f'<text x="{left}" y="{top - 15}">{title}</text>')
        for step in range(5):
            y = top + panel_height * step / 4
            elements.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_width}" y2="{y}" stroke="#ddd"/>')
            elements.append(f'<text x="5" y="{y + 4}">{high * (1 - step / 4):.3f}</text>')
        for epoch in range(0, max_epoch + 1, max(1, max_epoch // 5)):
            x = left + plot_width * epoch / max_epoch
            elements.append(f'<text x="{x}" y="{top + panel_height + 17}">{epoch}</text>')
        x_best = left + plot_width * best_epoch / max_epoch
        elements.append(f'<line x1="{x_best}" y1="{top}" x2="{x_best}" y2="{top + panel_height}" stroke="#777" stroke-dasharray="4 4"/>')
        for i, (key, color) in enumerate(series):
            points = ' '.join(f'{left + plot_width * r["epoch"] / max_epoch:.2f},{top + panel_height * (1 - r[key] / high):.2f}' for r in curve)
            elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8"/>')
            elements.append(f'<text x="{left + 390 + 150 * i}" y="{top - 15}" fill="{color}">{key}</text>')
    elements.append('</svg>')
    Path(path).write_text('\n'.join(elements))


def train_trajectory(name, features, rows, seed, out_dir, config, feature_identity):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'trajectory.jsonl').exists():
        raise ValueError('拒绝覆盖已有训练轨迹；请使用新目录')
    checkpoints = out / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train' and (name == 'G' or r['attack'])])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val' and (name == 'G' or r['attack'])])
    y = binary_labels([r['y'] for r in rows])
    xt, xv = [torch.from_numpy(np.asarray(features[index]).copy()) for index in [train, val]]
    yt, yv = [torch.tensor(y[index], dtype=torch.float32) for index in [train, val]]
    positive = int(yt.sum())
    negative = len(train) - positive
    if positive == 0 or negative == 0:
        raise ValueError('训练子集缺少类别')
    pos_weight = negative / positive
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = nn.Linear(xt.shape[1], 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    weighted_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    val_scores = np.lib.format.open_memmap(out / 'validation_scores.npy', mode='w+', dtype='float32',
                                         shape=(config['epochs'] + 1, len(val)))
    val_scores[:] = np.nan
    with (out / 'validation_manifest.jsonl').open('w') as f:
        for index in val:
            f.write(json.dumps({'id': rows[index]['id'], 'group_id': rows[index]['group_id'], 'y_A': int(y[index])}) + '\n')
    curve, best_loss, best_epoch, stale = [], float('inf'), None, 0
    simulated_stop, simulated_best = None, None
    start = time.perf_counter()
    with (out / 'trajectory.jsonl').open('w') as log:
        for epoch in range(config['epochs'] + 1):
            online_loss, gradient_sum, batches = 0., 0., 0
            if epoch:
                head.train()
                for batch in torch.randperm(len(train), generator=generator).split(config['head_batch']):
                    optimizer.zero_grad(set_to_none=True)
                    loss = weighted_loss(head(xt[batch]).squeeze(-1), yt[batch])
                    if not torch.isfinite(loss):
                        raise RuntimeError('非有限训练损失')
                    loss.backward()
                    gradient_sum += float(sum(p.grad.square().sum() for p in head.parameters()).sqrt())
                    optimizer.step()
                    online_loss += float(loss.detach()) * len(batch)
                    batches += 1
            head.eval()
            with torch.inference_mode():
                train_logits = head(xt).squeeze(-1)
                val_logits = head(xv).squeeze(-1)
                train_bce = float(nn.functional.binary_cross_entropy_with_logits(train_logits, yt))
                val_bce = float(nn.functional.binary_cross_entropy_with_logits(val_logits, yv))
                train_weighted = float(weighted_loss(train_logits, yt))
                val_weighted = float(weighted_loss(val_logits, yv))
                scores = val_logits.sigmoid().numpy()
            if not np.isfinite([train_bce, val_bce, train_weighted, val_weighted]).all():
                raise RuntimeError('非有限损失')
            threshold = fast_calibrate(y[val], scores, config['max_a_fpr'])
            predictions = (scores.astype(np.float64) >= threshold).astype(int)
            val_metrics = binary_metrics(y[val], predictions)
            val_scores[epoch] = scores
            val_scores.flush()
            is_best = epoch > 0 and val_bce < best_loss
            if epoch:
                if is_best:
                    best_loss, best_epoch, stale = val_bce, epoch, 0
                else:
                    stale += 1
                if simulated_stop is None and stale >= config['diagnostic_patience']:
                    simulated_stop, simulated_best = epoch, best_epoch
            record = {'epoch': epoch, 'train_bce': train_bce, 'val_bce': val_bce,
                      'train_weighted_bce': train_weighted, 'val_weighted_bce': val_weighted,
                      'online_weighted_bce': online_loss / len(train) if epoch else train_weighted,
                      'gradient_norm_mean': gradient_sum / batches if batches else 0.,
                      'learning_rate': optimizer.param_groups[0]['lr'],
                      'val_A_recall': val_metrics['A_recall'], 'val_A_fpr': val_metrics['A_fpr'],
                      'val_A_precision': val_metrics['A_precision'], 'val_threshold_A': threshold,
                      'best_so_far_epoch': best_epoch, 'stale_epochs': stale,
                      'elapsed_seconds': time.perf_counter() - start}
            checkpoint = {'epoch': epoch, 'name': name, 'seed': seed,
                          'state_dict': head.state_dict(), 'optimizer_state': optimizer.state_dict(),
                          'torch_rng_state': torch.get_rng_state(), 'shuffle_rng_state': generator.get_state(),
                          'hidden_size': xt.shape[1], 'output_size': 1, 'label_mapping': {'A': 1, 'nonA': 0},
                          'config': config, 'feature_identity': feature_identity,
                          'train_counts': {'A': positive, 'nonA': negative}, 'pos_weight': pos_weight,
                          'best_so_far_epoch': best_epoch, 'stale_epochs': stale, 'metrics': record}
            atomic_torch_save(checkpoint, checkpoints / f'epoch-{epoch:03d}.pt')
            if is_best:
                atomic_torch_save(checkpoint, out / 'best.pt')
            if epoch == config['epochs']:
                atomic_torch_save(checkpoint, out / 'final.pt')
            curve.append(record)
            log.write(json.dumps(record) + '\n')
            log.flush()
            (out / 'latest.json').write_text(json.dumps({'completed_epoch': epoch, 'checkpoint': f'checkpoints/epoch-{epoch:03d}.pt'}))
    last20 = curve[-min(20, config['epochs']):]
    summary = {'name': name, 'seed': seed, 'epochs_completed': config['epochs'],
               'train_n': len(train), 'val_n': len(val), 'best_epoch': best_epoch, 'best_val_bce': best_loss,
               'final_val_bce': curve[-1]['val_bce'], 'final_train_bce': curve[-1]['train_bce'],
               'initial_train_bce': curve[0]['train_bce'], 'initial_val_bce': curve[0]['val_bce'],
               'initial_train_weighted_bce': curve[0]['train_weighted_bce'],
               'final_train_weighted_bce': curve[-1]['train_weighted_bce'],
               'final_val_excess_percent': 100 * (curve[-1]['val_bce'] / best_loss - 1),
               'last20_val_bce_slope': float(np.polyfit(np.arange(len(last20)), [r['val_bce'] for r in last20], 1)[0]),
               'last20_train_weighted_bce_slope': float(np.polyfit(np.arange(len(last20)), [r['train_weighted_bce'] for r in last20], 1)[0]),
               'best_in_last20': best_epoch > config['epochs'] - 20,
               'simulated_patience10_stop_epoch': simulated_stop,
               'simulated_patience10_best_epoch': simulated_best,
               'elapsed_seconds': time.perf_counter() - start}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (out / 'trajectory.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    trajectory_svg(curve, best_epoch, out / 'trajectory.svg')
    print('完整轨迹 ' + json.dumps(summary), flush=True)
    return summary
