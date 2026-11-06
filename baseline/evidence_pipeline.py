"""将预测证据接入冻结large行为Probe；绝不以gold存在性路由或回填攻击原文。"""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from baseline.evidence_detector import gold_span, intersection
from baseline.modernbert_binary import binary_labels, binary_metrics, paired_bootstrap, predict_binary
from baseline.modernbert_capacity import replay_scores
from baseline.modernbert_data import digest, file_sha256, json_lines, tokenize_view
from baseline.modernbert_large_features import benchmark_checked
from baseline.modernbert_poc import cache_features, write_json
from baseline.modernbert_trajectory import fast_calibrate


def predicted_input_ids(question, current_draft, prediction, tokenizer):
    """此函数无gold/context/evaluator参数；未触发时不编码证据，不回填。"""
    if not prediction['detected']:
        if prediction['predicted_evidence']:
            raise ValueError('未触发分支却带有证据')
        return None
    if not prediction['predicted_evidence']:
        raise ValueError('触发分支却没有证据')
    fields = {'task': question, 'draft': current_draft, 'attack': prediction['predicted_evidence']}
    ids, _ = tokenize_view(fields, 'E', tokenizer, 4096)
    return ids


def build_inputs(rows, cases, predictions, tokenizer):
    row_index = {r['id']: i for i, r in enumerate(rows)}
    maps = {seed: np.full(len(rows), -1, dtype=np.int64) for seed in predictions}
    unique, tokens, lookup = [], [], {}
    for record in json_lines('artifacts/text_probe_v2/frozen/records.jsonl'):
        if record['id'] not in row_index:
            continue
        i = row_index[record['id']]
        case = cases[record['case_id']]
        for seed, per_case in predictions.items():
            prediction = per_case[case['id']]
            ids = predicted_input_ids(case['question'], record['input']['current_draft'], prediction, tokenizer)
            if ids is None:
                continue
            h = digest(ids)
            if h not in lookup:
                lookup[h] = len(unique)
                unique.append({'id': f'predicted-evidence-{h}', 'E_hash': h})
                tokens.append(ids)
            maps[seed][i] = lookup[h]
    for seed, mapping in maps.items():
        assert all((mapping[i] >= 0) == predictions[seed][r['case_id']]['detected'] for i, r in enumerate(rows))
    return unique, tokens, maps


def encode_predicted(unique, sequences, source_rows, source_e, tokenizer, model_info, out, contract):
    out.mkdir(parents=True, exist_ok=True)
    width = source_e.shape[1]
    features = np.lib.format.open_memmap(out / 'E_features.npy', mode='w+', dtype='float32', shape=(len(unique), width))
    done = np.lib.format.open_memmap(out / 'E_done.npy', mode='w+', dtype='bool', shape=(len(unique),))
    features[:] = np.nan
    done[:] = False
    # 仅按完全一致的输入token哈希复用确定性函数缓存，不影响路由或证据选择。
    existing = {r['E_hash']: i for i, r in enumerate(source_rows) if r.get('E_hash')}
    reused = 0
    for i, row in enumerate(unique):
        if row['E_hash'] in existing:
            features[i] = source_e[existing[row['E_hash']]]
            done[i] = True
            reused += 1
    features.flush()
    done.flush()
    write_json(out / 'E_cache.json', {'contract_hash': digest(contract), 'shape': [len(unique), width], 'dtype': 'float32', 'view': 'E'})
    missing = [ids for ids, complete in zip(sequences, done) if not complete]
    started = time.perf_counter()
    if missing:
        torch.cuda.set_per_process_memory_fraction(.5)
        cfg = AutoConfig.from_pretrained(model_info['local_path'], local_files_only=True)
        cfg.reference_compile = False
        model = AutoModel.from_pretrained(model_info['local_path'], config=cfg, local_files_only=True,
            dtype=torch.float32, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        batches = benchmark_checked(model, tokenizer, {'G': [], 'E': missing}, out)
        features = cache_features(model, tokenizer, unique, {'E': sequences}, 'E', out, contract, batches)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    assert np.isfinite(features).all() and done.all()
    info = {'unique_predicted_inputs': len(unique), 'exact_token_cache_hits': reused,
            'new_encoder_inputs': len(missing), 'elapsed_seconds': time.perf_counter() - started,
            'cache_reuse': 'only exact token hash equality; never change predicted text or routing'}
    write_json(out / 'encoding_summary.json', info)
    return features, info


def evidence_scores(head_path, features, mapping):
    checkpoint = torch.load(head_path, map_location='cpu', weights_only=True)
    head = torch.nn.Linear(checkpoint['hidden_size'], 1)
    head.load_state_dict(checkpoint['state_dict'])
    head.eval()
    output = np.full(len(mapping), np.nan, dtype=np.float32)
    active = np.flatnonzero(mapping >= 0)
    with torch.inference_mode():
        for offset in range(0, len(active), 256):
            idx = active[offset:offset + 256]
            x = np.asarray(features[mapping[idx]]).copy()
            output[idx] = head(torch.from_numpy(x)).squeeze(-1).sigmoid().numpy()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--detector-dir', required=True)
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP'):
        raise SystemExit('需资源限额wrapper')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    out, detector = Path(args.run_dir), Path(args.detector_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'pipeline_config.json').exists():
        raise ValueError('不能覆盖既有下游运行')
    assert (detector / 'completed.json').exists()
    assert json.loads((detector / 'run-resources/exit.json').read_text())['exit_code'] == 0
    source = Path('artifacts/modernbert_oracle_poc/large-features-20261105-v2')
    probes = Path('artifacts/modernbert_oracle_poc/large-vs-base-20261105')
    rows = list(json_lines(source / 'manifest.jsonl'))
    cases = {c['id']: c for c in json_lines('artifacts/text_probe_v2/frozen/resolved_cases.jsonl')}
    model_info = json.loads(Path('artifacts/modernbert_oracle_poc/model-large.json').read_text())
    predictions = {seed: {p['case_id']: p for p in json_lines(detector / f'seed-{seed}/predictions.jsonl')} for seed in [42, 43, 44]}
    for per_case in predictions.values():
        assert set(per_case) == set(cases)
        for case_id, prediction in per_case.items():
            message = cases[case_id]['context']['ordered_messages'][prediction['message_index']]
            assert message['role'] == 'tool'
            if prediction['detected']:
                assert prediction['predicted_evidence'] == message['content'][prediction['start']:prediction['end']]
            else:
                assert prediction['predicted_evidence'] == ''
    config = {'model': model_info, 'detector_dir': str(detector.resolve()), 'probe_dir': str(probes.resolve()),
              'source_dir': str(source.resolve()), 'protocol_sha256': file_sha256(out / 'protocol.json'),
              'detector_predictions_sha256': {str(seed): file_sha256(detector / f'seed-{seed}/predictions.jsonl') for seed in predictions},
              'probe_checkpoints_sha256': {f'{name}/{seed}': file_sha256(probes / 'large' / f'{name}-seed-{seed}/best.pt')
                                         for name in ['G', 'E'] for seed in predictions},
              'source_manifest_sha256': file_sha256(source / 'manifest.jsonl'),
              'source_features_sha256': {v: file_sha256(source / f'{v}_features.npy') for v in ['G', 'E']},
              'code_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in ['evidence_pipeline.py', 'evidence_detector.py', 'modernbert_data.py']},
              'metric_scope': 'same eligible block dataset; case detector threshold uses attack presence, behavior threshold uses A risk',
              'behavior_heads_frozen': True}
    write_json(out / 'pipeline_config.json', config)
    tokenizer = AutoTokenizer.from_pretrained(model_info['local_path'], local_files_only=True)
    unique, sequences, maps = build_inputs(rows, cases, predictions, tokenizer)
    write_json(out / 'predicted_inputs.json', unique)
    for seed, mapping in maps.items():
        np.save(out / f'row_mapping_seed-{seed}.npy', mapping)
    source_g = np.load(source / 'G_features.npy', mmap_mode='r')
    source_e = np.load(source / 'E_features.npy', mmap_mode='r')
    contract = {'config': config, 'unique_inputs_hash': digest(unique)}
    write_json(out / 'predicted_cache_contract.json', contract)
    predicted_features, encoding = encode_predicted(unique, sequences, rows, source_e, tokenizer, model_info,
                                                   out / 'predicted_E_cache', contract)
    del sequences
    old_results = json.loads((probes / 'results.json').read_text())
    original_predictions = {(p['seed'], p['configuration'], p['id']): p for p in json_lines(probes / 'test_predictions.jsonl') if p['model'] == 'large'}
    y = binary_labels([r['y'] for r in rows])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test'])
    enabled_gold = np.array([r['attack'] for r in rows])
    results = {'seeds': {}, 'encoding': encoding, 'oracle_reference': str(probes)}
    with (out / 'test_predictions.jsonl').open('w') as f:
        for seed in predictions:
            g = replay_scores(probes / 'large' / f'G-seed-{seed}', source_g, rows, 'G')
            e_gold = replay_scores(probes / 'large' / f'E-seed-{seed}', source_e, rows, 'E')
            e_pred = evidence_scores(probes / 'large' / f'E-seed-{seed}/best.pt', predicted_features, maps[seed])
            oracle, predicted = g.copy(), g.copy()
            oracle[enabled_gold] = np.maximum(g[enabled_gold], e_gold[enabled_gold])
            enabled_pred = maps[seed] >= 0
            predicted[enabled_pred] = np.maximum(g[enabled_pred], e_pred[enabled_pred])
            scores = {'G': g, 'Oracle': oracle, 'Predicted': predicted}
            thresholds = {'G': old_results['seeds'][str(seed)]['large']['M0']['threshold_A'],
                          'Oracle': old_results['seeds'][str(seed)]['large']['M2']['threshold_A'],
                          'Predicted': fast_calibrate(y[val], predicted[val], .05)}
            entries, decisions = {}, {}
            for name, score in scores.items():
                threshold = thresholds[name]
                decision = predict_binary(score[test], threshold)
                decisions[name] = decision
                strata = {}
                for field in ['domain', 'variant', 'label_source']:
                    group = np.array([rows[i][field] for i in test])
                    strata[field] = {str(k): binary_metrics(y[test][group == k], decision[group == k]) for k in sorted(set(group))}
                entries[name] = {'threshold_A': threshold, 'test': binary_metrics(y[test], decision),
                                 'validation': binary_metrics(y[val], predict_binary(score[val], threshold)), 'strata': strata}
                if name in ['G', 'Oracle']:
                    old_name = 'M0' if name == 'G' else 'M2'
                    assert decision.tolist() == [original_predictions[(seed, old_name, rows[i]['id'])]['prediction_A'] for i in test]
                for j, index in enumerate(test):
                    row = rows[index]
                    prediction = predictions[seed][row['case_id']]
                    gold = gold_span(cases[row['case_id']])
                    overlap = intersection(prediction, gold) if prediction['detected'] else 0
                    coverage = overlap / (gold[2] - gold[1]) if gold else None
                    f.write(json.dumps({'seed': seed, 'configuration': name, 'id': row['id'], 'case_id': row['case_id'],
                                        'group_id': row['group_id'], 'y_A': int(y[index]), 'prediction_A': int(decision[j]),
                                        'score_A': float(score[index]), 'detector_positive': prediction['detected'],
                                        'detector_score': prediction['score_attack'], 'gold_coverage_audit_only': coverage}) + '\n')
            reasons = Counter()
            missed_ids = []
            for j, i in enumerate(test):
                if y[i] != 1 or decisions['Oracle'][j] != 1 or decisions['Predicted'][j] == 1:
                    continue
                prediction = predictions[seed][rows[i]['case_id']]
                gold = gold_span(cases[rows[i]['case_id']])
                overlap = intersection(prediction, gold) if prediction['detected'] else 0
                if not prediction['detected']:
                    reason = 'detector_did_not_enable_branch'
                elif overlap == 0:
                    reason = 'wrong_evidence_span'
                elif overlap < gold[2] - gold[1]:
                    reason = 'partial_evidence_span'
                else:
                    reason = 'full_gold_covered_but_behavior_threshold_not_met'
                reasons[reason] += 1
                missed_ids.append({'id': rows[i]['id'], 'reason': reason})
            entries['oracle_hit_predicted_miss'] = {'counts': dict(reasons), 'records': missed_ids,
                'caveat': '诊断分组，不是严格因果归因；Predicted的聚合阈值重新用val校准，可能与Oracle不同'}
            boot = paired_bootstrap(y[test], {'M0': decisions['G'], 'M1': decisions['Oracle'], 'M2': decisions['Predicted']},
                                    np.array([rows[i]['group_id'] for i in test]), seed, 500)
            boot['percentile_95'] = {k.replace('M2', 'Predicted').replace('M1', 'Oracle').replace('M0', 'G'): v
                                     for k, v in boot['percentile_95'].items()}
            entries['bootstrap'] = boot
            results['seeds'][str(seed)] = entries
    results['means'] = {name: {metric: statistics.mean(r[name]['test'][metric] for r in results['seeds'].values())
                              for metric in ['A_recall', 'A_fpr', 'A_precision']} for name in ['G', 'Oracle', 'Predicted']}
    write_json(out / 'results.json', results)
    detector_results = json.loads((detector / 'results.json').read_text())
    lines = ['# 预测攻击证据接入large Probe', '',
             '本轮只新增攻击存在检测＋段落定位，不按攻击类型路由。检测端冻结ModernBERT-base，后端large G/E行为头完全冻结。',
             '预测分支只读检测器从可见tool材料输出的原文，不读取attack_str元数据；没有gold回填。', '',
             '## 检测器测试集', '', '| seed | 攻击存在召回 | 干净case误报 | 攻击原文完整覆盖 | 与gold完全一致 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for seed, d in detector_results.items():
        m = d['metrics']['test']
        lines.append(f'| {seed} | {m["attack_recall"]:.2%} | {m["clean_fpr"]:.2%} | {m["full_gold_coverage_rate"]:.2%} | {m["exact_evidence_rate"]:.2%} |')
    lines += ['', '攻击存在指标是case级，不是主模型中招的A类指标。攻击被抵抗的case仍是攻击存在正例。', '',
              '## 行为检测测试集：三seed均值', '', '| 方案 | A召回 | A误报率 | A精确率 |', '| --- | ---: | ---: | ---: |']
    for name, m in results['means'].items():
        lines.append(f'| {name} | {m["A_recall"]:.2%} | {m["A_fpr"]:.2%} | {m["A_precision"]:.2%} |')
    lines += ['', 'G为通用头；Oracle为已知攻击证据组合；Predicted为预测证据组合。Oracle/G重放与既有结果逐点一致。', '',
              '## 边界与审计', '',
              '- 检测器在全部case的既有train/val/test上训练/校准/评估；行为指标仍使用相同4110/1078/1883条有效块，没有补未知行为标签。',
              '- 候选生成只按空行和192token窗口/96stride，不使用gold边界；输入只有用户任务和候选材料，不给位置、case或split编码。',
              '- 攻击原文在现有945个有攻击case中都可唯一精确定位于tool材料末尾，段落定位偏容易；不覆盖任意位置、正常引用攻击、编码隐写等困难情况。',
              '- 检测器只负责是否存在攻击及证据位置，不负责A/非A；行为头仍可能漏掉主模型中招。',
              '- 两层阈值分别在val校准：检测器控制干净case误报≤5%，聚合行为头控制A块级误报≤5%；不能把两种误报率混同。',
              '- Predicted评估允许clean误触发E分支，允许错误证据，不按gold过滤。',
              '- 重用已有E特征仅限预测输入token哈希完全一致；不更改预测文本或启用mask，不是gold回填。',
              '- 既有large E是oracle文本训练，预测段落可能有分布偏移。本轮先不重新训练它，以隔离证据组件影响。',
              '- 相同test已看过，结果是探索性的；攻击方式分析未用来添加特定test攻击训练样本。',
              '- 新增漏检诊断与配对来源组bootstrap见results.json；诊断标签不等同严格因果归因。',
              f'- 预测证据独特输入{encoding["unique_predicted_inputs"]}个，精确缓存命中{encoding["exact_token_cache_hits"]}个，新编码{encoding["new_encoder_inputs"]}个。', '']
    (out / 'report.md').write_text('\n'.join(lines))
    write_json(out / 'completed.json', {'config_hash': digest(config), 'elapsed_seconds': time.perf_counter() - started})
    print(f'预测证据下游实验完成：{out / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
