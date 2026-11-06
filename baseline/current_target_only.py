"""S3：只加当前待发布主模型输出，不读额外的模型输出历史；与冻结S0/S2对比。"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from baseline.evidence_pipeline import evidence_scores
from baseline.modernbert_binary import binary_labels, binary_metrics, predict_binary
from baseline.modernbert_capacity import TRAIN_CONFIG, replay_scores
from baseline.modernbert_data import digest, file_sha256, json_lines, tokenize_view, view_fields
from baseline.modernbert_large_features import benchmark_checked
from baseline.modernbert_poc import cache_features, write_json
from baseline.modernbert_trajectory import fast_calibrate, train_trajectory
from baseline.target_text_ablation import materialize, paired_intervals
from scripts.render_checked_trajectories import render

REFERENCE = Path('artifacts/modernbert_oracle_poc/target-text-ablation-20261106')
BASELINE = Path('artifacts/modernbert_oracle_poc/predicted-evidence-20261105')
VIEWS = ['G', 'E_oracle', 'E_pred_42', 'E_pred_43', 'E_pred_44']


def current_only_tokens(record, case, tokenizer, view, evidence=None):
    """不访问history_pairs；supervision只读取当前target_output。"""
    if view not in ['G', 'E']:
        raise ValueError('未知视图')
    fields = view_fields(record, case)
    if view == 'E' and evidence is not None:
        fields['attack'] = evidence
    current = record['supervision']['target_output']
    current_ids = tokenizer.encode(current, add_special_tokens=False)
    extra = tokenizer.encode('\n[TARGET_CURRENT]\n', add_special_tokens=False) + current_ids
    if len(extra) >= 4094:
        raise ValueError('当前输出占满预算，不静默裁剪')
    base, audit = tokenize_view(fields, view, tokenizer, 4096 - len(extra))
    ids = base[:-1] + extra + [tokenizer.sep_token_id]
    assert len(ids) <= 4096
    audit.update(arm='S3', total_length=len(ids), current_target_tokens=len(current_ids), history_field_included=False)
    return ids, audit


def build_inputs(out, tokenizer):
    baseline_cfg = json.loads((BASELINE / 'pipeline_config.json').read_text())
    source = Path(baseline_cfg['source_dir'])
    detector = Path(baseline_cfg['detector_dir'])
    rows = list(json_lines(source / 'manifest.jsonl'))
    cases = {c['id']: c for c in json_lines('artifacts/text_probe_v2/frozen/resolved_cases.jsonl')}
    predictions = {seed: {p['case_id']: p for p in json_lines(detector / f'seed-{seed}/predictions.jsonl')} for seed in [42, 43, 44]}
    indices = {r['id']: i for i, r in enumerate(rows)}
    maps = {view: np.full(len(rows), -1, dtype=np.int64) for view in VIEWS}
    unique, sequences, lookup = [], [], {}
    lengths, conflicts, trimmed = defaultdict(list), defaultdict(lambda: defaultdict(set)), Counter()
    with (out / 'input_audit.jsonl').open('w') as log:
        for record in json_lines('artifacts/text_probe_v2/frozen/records.jsonl'):
            if record['id'] not in indices:
                continue
            i, case = indices[record['id']], cases[record['case_id']]
            assert case['split'] == rows[i]['split'] and case['group_id'] == rows[i]['group_id']
            for view in VIEWS:
                evidence = None
                if view == 'E_oracle' and not rows[i]['attack']:
                    continue
                if view.startswith('E_pred_'):
                    prediction = predictions[int(view.rsplit('_', 1)[1])][case['id']]
                    if not prediction['detected']:
                        assert not prediction['predicted_evidence']
                        continue
                    evidence = prediction['predicted_evidence']
                    message = case['context']['ordered_messages'][prediction['message_index']]
                    assert message['role'] == 'tool' and evidence == message['content'][prediction['start']:prediction['end']]
                ids, audit = current_only_tokens(record, case, tokenizer, 'G' if view == 'G' else 'E', evidence)
                h = digest(ids)
                if h not in lookup:
                    lookup[h] = len(unique)
                    unique.append({'id': 'S3-' + h, 'G_hash': h})
                    sequences.append(ids)
                maps[view][i] = lookup[h]
                conflicts[view][h].add(rows[i]['y'])
                lengths[view].append(audit['total_length'])
                trimmed[view] += int(audit['trimmed_tokens'] > 0)
                log.write(json.dumps({'id': record['id'], 'view_name': view, 'token_hash': h, **audit}) + '\n')
    assert (maps['G'] >= 0).all()
    assert (maps['E_oracle'] >= 0).tolist() == [r['attack'] for r in rows]
    for seed in [42, 43, 44]:
        expected = [predictions[seed][r['case_id']]['detected'] for r in rows]
        assert (maps[f'E_pred_{seed}'] >= 0).tolist() == expected
        previous = np.load(REFERENCE / 'features' / f'S2_E_pred_{seed}_mapping.npy')
        assert np.array_equal(maps[f'E_pred_{seed}'] >= 0, previous >= 0)
    for view, mapping in maps.items():
        np.save(out / f'{view}_mapping.npy', mapping)
    with (out / 'manifest.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(out / 'unique_inputs.json', unique)
    audit = {'eligible_rows': len(rows), 'unique_encoder_inputs': len(unique), 'history_field_included': False,
             'scope': '当前输出可见的发布前审查，不是提前预测；保留原有聊天上下文',
             'counts': dict(Counter(f'{r["split"]}/{r["y"]}' for r in rows)),
             'lengths': {v: {'n': len(x), 'max': max(x), 'mean': float(np.mean(x)), 'trimmed': trimmed[v]} for v, x in lengths.items()},
             'conflicting_input_hashes': {v: sum(len(labels) > 1 for labels in hashes.values()) for v, hashes in conflicts.items()}}
    write_json(out / 'data_audit.json', audit)
    print('S3输入审计 ' + json.dumps(audit, ensure_ascii=False), flush=True)
    return rows, unique, sequences, maps


def extract(run):
    out = run / 'features'
    out.mkdir(parents=True, exist_ok=True)
    model_info = json.loads(Path('artifacts/modernbert_oracle_poc/model-large.json').read_text())
    baseline_cfg = json.loads((BASELINE / 'pipeline_config.json').read_text())
    config = {'model': model_info, 'serialization': 'v2-plus-current-target-only-v1', 'history_field_included': False,
              'scope': 'post_generation_pre_release', 'pooling': 'last_hidden_nonpadding_mean_fp32',
              'protocol_sha256': file_sha256(run / 'protocol.json'),
              'sources': {'records': file_sha256('artifacts/text_probe_v2/frozen/records.jsonl'),
                          'cases': file_sha256('artifacts/text_probe_v2/frozen/resolved_cases.jsonl')},
              'detector_predictions_sha256': baseline_cfg['detector_predictions_sha256'],
              'code_sha256': {name: file_sha256(Path(name)) for name in [
                  'baseline/current_target_only.py', 'baseline/modernbert_data.py', 'baseline/modernbert_large_features.py', 'baseline/modernbert_poc.py']}}
    detector = Path(baseline_cfg['detector_dir'])
    for seed, h in config['detector_predictions_sha256'].items():
        assert file_sha256(detector / f'seed-{seed}/predictions.jsonl') == h
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != config:
        raise ValueError('不可混用不同配置的缓存')
    write_json(out / 'config.json', config)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_info['local_path'], local_files_only=True)
    rows, unique, sequences, maps = build_inputs(out, tokenizer)
    contract = {'config': config, 'manifest_hash': digest(rows), 'unique_inputs_hash': digest(unique)}
    write_json(out / 'cache_contract.json', contract)
    torch.cuda.set_per_process_memory_fraction(.5)
    cfg = AutoConfig.from_pretrained(model_info['local_path'], local_files_only=True)
    cfg.reference_compile = False
    model = AutoModel.from_pretrained(model_info['local_path'], config=cfg, local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    batches = (json.loads((out / 'batch_benchmark.json').read_text())['selected_batches'] if (out / 'batch_benchmark.json').exists()
               else benchmark_checked(model, tokenizer, {'G': sequences, 'E': []}, out))
    cache_features(model, tokenizer, unique, {'G': sequences}, 'G', out, contract, batches)
    write_json(out / 'completed.json', {'config_hash': digest(config), 'elapsed_seconds': time.perf_counter() - started,
        'unique_inputs': len(unique), 'hidden_size': cfg.hidden_size, 'trainable_backbone_parameters': 0})
    print(f'S3特征完成：{out}', flush=True)


def evaluate(run, rows, vectors, maps):
    old_results = json.loads((REFERENCE / 'results.json').read_text())
    old_logs = list(json_lines(REFERENCE / 'test_predictions.jsonl'))
    mechanisms = {r['id']: r['mechanism'] for r in json_lines('artifacts/modernbert_oracle_poc/attack-recall-analysis-20261105/A_cases.jsonl')}
    y = binary_labels([r['y'] for r in rows])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test'])
    gold_mask = np.array([r['attack'] for r in rows])
    result = {'seeds': {}, 'bootstrap': {}}
    with (run / 'test_predictions.jsonl').open('w') as f:
        for seed in TRAIN_CONFIG['seeds']:
            entries = {arm: old_results['seeds'][str(seed)][arm] for arm in ['S0', 'S2']}
            decisions = {}
            for arm in ['S0', 'S2']:
                records = [r for r in old_logs if r['seed'] == seed and r['arm'] == arm]
                for record in records:
                    f.write(json.dumps(record) + '\n')
                selected = [r for r in records if r['configuration'] == 'Predicted']
                assert [r['id'] for r in selected] == [rows[i]['id'] for i in test]
                decisions[arm] = np.array([r['prediction_A'] for r in selected])
            g = replay_scores(run / 'S3' / f'G-seed-{seed}', materialize(vectors, maps['G']), rows, 'G')
            e = replay_scores(run / 'S3' / f'E-seed-{seed}', materialize(vectors, maps['E_oracle']), rows, 'E')
            pred_e = evidence_scores(run / 'S3' / f'E-seed-{seed}/best.pt', vectors, maps[f'E_pred_{seed}'])
            enabled = maps[f'E_pred_{seed}'] >= 0
            oracle, predicted = g.copy(), g.copy()
            oracle[gold_mask] = np.maximum(g[gold_mask], e[gold_mask])
            predicted[enabled] = np.maximum(g[enabled], pred_e[enabled])
            entries['S3'] = {}
            for name, scores in [('G', g), ('Oracle', oracle), ('Predicted', predicted)]:
                threshold = fast_calibrate(y[val], scores[val], .05)
                pred = predict_binary(scores[test], threshold)
                strata = {}
                for field in ['domain', 'label_source', 'variant']:
                    group = np.array([rows[i][field] for i in test])
                    strata[field] = {str(k): binary_metrics(y[test][group == k], pred[group == k]) for k in sorted(set(group))}
                by_mechanism = {}
                for mechanism in sorted(set(mechanisms.values())):
                    selected = np.array([j for j, i in enumerate(test) if mechanisms.get(rows[i]['id']) == mechanism])
                    if len(selected):
                        by_mechanism[mechanism] = {'A_blocks': len(selected), 'A_recall': float(pred[selected].mean())}
                entries['S3'][name] = {'threshold_A': threshold, 'test': binary_metrics(y[test], pred),
                    'validation': binary_metrics(y[val], predict_binary(scores[val], threshold)),
                    'strata': strata, 'A_mechanisms': by_mechanism}
                if name == 'Predicted':
                    decisions['S3'] = pred
                for j, index in enumerate(test):
                    row = rows[index]
                    f.write(json.dumps({'arm': 'S3', 'seed': seed, 'configuration': name, 'id': row['id'],
                        'case_id': row['case_id'], 'group_id': row['group_id'], 'y_A': int(y[index]),
                        'score_A': float(scores[index]), 'prediction_A': int(pred[j]),
                        'detector_positive': bool(enabled[index]), 'current_target_visible': True,
                        'extra_target_history_visible': False}) + '\n')
            # 复用已测试的三臂配对bootstrap，仅内部别名映射，输出恢复实际方案名。
            boot = paired_intervals(y[test], {'S0': decisions['S0'], 'S1': decisions['S2'], 'S2': decisions['S3']},
                                    np.array([rows[i]['group_id'] for i in test]), seed)
            rename = {'S1-minus-S0': 'S2-minus-S0', 'S2-minus-S0': 'S3-minus-S0', 'S2-minus-S1': 'S3-minus-S2'}
            boot['percentile_95'] = {rename[k.split('/')[0]] + '/' + k.split('/')[1]: v for k, v in boot['percentile_95'].items()}
            result['seeds'][str(seed)] = entries
            result['bootstrap'][str(seed)] = boot
    result['means'] = {arm: {name: {metric: statistics.mean(e[arm][name]['test'][metric] for e in result['seeds'].values())
        for metric in ['A_recall', 'A_fpr', 'A_precision', 'A_f1']} for name in ['G', 'Oracle', 'Predicted']} for arm in ['S0', 'S2', 'S3']}
    write_json(run / 'results.json', result)
    return result


def report(run, results, summaries, elapsed):
    lines = ['# S3：只读当前待发布主模型输出，不增加输出块历史', '',
        'S0无额外主模型文本；S2有历史与当前块；S3只加当前块。保留原用户任务、草稿和G原有聊天上下文/E证据。',
        'S2/S3均为生成后、发布前审查，不是提前预测；未实施在线输出缓冲或工具动作拦截。', '',
        '## 主要结果：实际预测证据，三seed均值', '',
        '| 方案 | A召回 | A误报率 | A精确率 | A-F1 |', '| --- | ---: | ---: | ---: | ---: |']
    for arm in ['S0', 'S2', 'S3']:
        m = results['means'][arm]['Predicted']
        lines.append(f'| {arm} | {m["A_recall"]:.2%} | {m["A_fpr"]:.2%} | {m["A_precision"]:.2%} | {m["A_f1"]:.4f} |')
    lines += ['', '## 单头与oracle诊断', '', '| 方案 | G召回 | G误报 | Oracle召回 | Oracle误报 |', '| --- | ---: | ---: | ---: | ---: |']
    for arm in ['S0', 'S2', 'S3']:
        g, e = results['means'][arm]['G'], results['means'][arm]['Oracle']
        lines.append(f'| {arm} | {g["A_recall"]:.2%} | {g["A_fpr"]:.2%} | {e["A_recall"]:.2%} | {e["A_fpr"]:.2%} |')
    lines += ['', '## S3完整训练轨迹', '', '| 头 | seed | 最佳epoch | 最佳val BCE | 最终val BCE |', '| --- | --- | ---: | ---: | ---: |']
    for s in summaries:
        lines.append(f'| {s["name"]} | {s["seed"]} | {s["best_epoch"]} | {s["best_val_bce"]:.4f} | {s["final_val_bce"]:.4f} |')
    lines += ['', '## 固定条件与边界', '',
        '- 不读取history_pairs，不添加TARGET_HISTORY字段。G保留原有聊天中的assistant消息，不是删除所有历史对话。',
        '- 只额外读取当前target_output，不读取标签、参考答案、判定说明或完整未来回答。',
        '- 冻结同一ModernBERT-large FP32与mean pooling；数据、分组、意图检测器、预测片段与路由mask均不变。',
        '- G使用4110条train；E仍用2071条oracle攻击子集训练和验证选点，主结果接入预测证据。',
        '- 六个头各训练100epoch，AdamW lr1e-3、weight_decay1e-4、batch128；按未加权val BCE选best，test不选点。',
        '- 阈值只在val A误报≤5%下最大化召回，实际test误报率如实报告；按来源组配对bootstrap见results.json。',
        '- S0/S2结果与阈值直接复用，不重新择优；同一test已多次查看，所有结论仍是探索性的。',
        '- 每epoch权重、optimizer/RNG、验证预测、JSONL/CSV轨迹保留；查看trajectory.checked.svg，旧绘图原件仅作留档。',
        f'- S3头训练评估逻辑耗时{elapsed:.2f}秒，特征提取另计。', '']
    (run / 'report.md').write_text('\n'.join(lines))


def train(run):
    source = run / 'features'
    assert json.loads((source / 'run-resources/exit.json').read_text())['exit_code'] == 0
    assert (REFERENCE / 'verification.json').exists()
    if (run / 'config.json').exists():
        raise ValueError('不能覆盖既有训练')
    started = time.perf_counter()
    fcfg = json.loads((source / 'config.json').read_text())
    assert json.loads((source / 'completed.json').read_text())['config_hash'] == digest(fcfg)
    rows = list(json_lines(source / 'manifest.jsonl'))
    assert rows == list(json_lines(REFERENCE / 'features/manifest.jsonl'))
    maps = {view: np.load(source / f'{view}_mapping.npy') for view in VIEWS}
    vectors = np.load(source / 'G_features.npy', mmap_mode='r')
    assert np.load(source / 'G_done.npy').all() and np.isfinite(vectors).all()
    config = {'training': TRAIN_CONFIG, 'feature_config': fcfg, 'feature_sha256': file_sha256(source / 'G_features.npy'),
        'manifest_hash': digest(rows), 'protocol_sha256': file_sha256(run / 'protocol.json'),
        'reference_results_sha256': file_sha256(REFERENCE / 'results.json'),
        'reference_predictions_sha256': file_sha256(REFERENCE / 'test_predictions.jsonl'),
        'code_sha256': {name: file_sha256(Path(name)) for name in [
            'baseline/current_target_only.py', 'baseline/modernbert_trajectory.py', 'baseline/target_text_ablation.py', 'scripts/render_checked_trajectories.py']}}
    write_json(run / 'config.json', config)
    g, e = materialize(vectors, maps['G']), materialize(vectors, maps['E_oracle'])
    summaries = []
    for seed in TRAIN_CONFIG['seeds']:
        for name, features in [('G', g), ('E', e)]:
            identity = {'arm': 'S3', 'view': name, 'feature_sha256': config['feature_sha256'], 'current_target_visible': True,
                        'extra_target_history_visible': False, 'mapping_sha256': file_sha256(source / f'{"G" if name == "G" else "E_oracle"}_mapping.npy')}
            folder = run / 'S3' / f'{name}-seed-{seed}'
            summary = train_trajectory(name, features, rows, seed, folder, TRAIN_CONFIG, identity)
            curve = list(json_lines(folder / 'trajectory.jsonl'))
            render(curve, summary['best_epoch'], f'S3 / {name} / seed {seed}', folder / 'trajectory.checked.svg')
            summaries.append(summary)
    write_json(run / 'trajectory_summary.json', summaries)
    results = evaluate(run, rows, vectors, maps)
    elapsed = time.perf_counter() - started
    report(run, results, summaries, elapsed)
    write_json(run / 'completed.json', {'config_hash': digest(config), 'elapsed_seconds': elapsed})
    print(f'S3对照已完成：{run / "report.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--stage', choices=['features', 'train'], required=True)
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP'):
        raise SystemExit('必须通过受限wrapper')
    if args.stage == 'train' and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('头训练仅CPU，需禁用CUDA可见性')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.stage == 'features':
        extract(Path(args.run_dir))
    else:
        train(Path(args.run_dir))


if __name__ == '__main__':
    main()
