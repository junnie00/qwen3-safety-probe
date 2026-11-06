"""主模型文本消融：S1只读过去target，S2允许当前待发布target；共享冻结large编码。"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from baseline.modernbert_data import digest, file_sha256, json_lines, tokenize_view, view_fields
from baseline.modernbert_large_features import benchmark_checked
from baseline.modernbert_poc import cache_features, write_json

ARMS = ('S1', 'S2')
HISTORY_TOKENS = 256


def target_text_tokens(record, case, tokenizer, arm, view, evidence=None):
    """S1在函数级不读取supervision；S2仅白名单读取target_output。"""
    if arm not in ARMS or view not in ['G', 'E']:
        raise ValueError('未知实验组/视图')
    fields = view_fields(record, case)
    if view == 'E' and evidence is not None:
        fields['attack'] = evidence
    history_text = ''.join(pair['target'] for pair in record['input']['history_pairs'])
    history_ids = tokenizer.encode(history_text, add_special_tokens=False)
    history_tail = history_ids[-HISTORY_TOKENS:]
    extra = tokenizer.encode('\n[TARGET_HISTORY]\n', add_special_tokens=False) + history_tail
    current_length = 0
    if arm == 'S2':
        current = record['supervision']['target_output']
        current_ids = tokenizer.encode(current, add_special_tokens=False)
        current_length = len(current_ids)
        extra += tokenizer.encode('\n[TARGET_CURRENT]\n', add_special_tokens=False) + current_ids
    if len(extra) >= 4094:
        raise ValueError('主模型文本占满预算，不静默截断')
    base, audit = tokenize_view(fields, view, tokenizer, 4096 - len(extra))
    ids = base[:-1] + extra + [tokenizer.sep_token_id]
    assert len(ids) <= 4096 and ids[0] == tokenizer.cls_token_id and ids[-1] == tokenizer.sep_token_id
    audit.update(arm=arm, total_length=len(ids), history_original_tokens=len(history_ids),
                 history_retained_tokens=len(history_tail), current_target_tokens=current_length)
    return ids, audit


def build_target_inputs(out, tokenizer, source, detector):
    out = Path(out)
    rows = list(json_lines(source / 'manifest.jsonl'))
    cases = {c['id']: c for c in json_lines('artifacts/text_probe_v2/frozen/resolved_cases.jsonl')}
    indices = {r['id']: i for i, r in enumerate(rows)}
    predictions = {seed: {p['case_id']: p for p in json_lines(detector / f'seed-{seed}/predictions.jsonl')} for seed in [42, 43, 44]}
    maps = {arm: {view: np.full(len(rows), -1, dtype=np.int64) for view in ['G', 'E_oracle', 'E_pred_42', 'E_pred_43', 'E_pred_44']} for arm in ARMS}
    unique, sequences, lookup = [], [], {}
    temporal, conflicts, lengths = {}, defaultdict(lambda: defaultdict(set)), defaultdict(list)
    trim_counts = Counter()
    audited_records, max_initial = 0, 0

    def insert(ids):
        h = digest(ids)
        if h not in lookup:
            lookup[h] = len(unique)
            unique.append({'id': 'target-view-' + h, 'G_hash': h})
            sequences.append(ids)
        return lookup[h], h

    with (out / 'input_audit.jsonl').open('w') as log:
        for record in json_lines('artifacts/text_probe_v2/frozen/records.jsonl'):
            cid = record['case_id']
            block = int(record['id'].rsplit('/block-', 1)[1])
            history = [p['target'] for p in record['input']['history_pairs']]
            if cid not in temporal:
                assert block == 0
                temporal[cid] = {'expected': history.copy(), 'next': 0}
                max_initial = max(max_initial, len(history))
            state = temporal[cid]
            assert block == state['next'] and history == state['expected'], '时间边界不一致'
            # 当前target只加入下一块所需的审计状态，不传给S1序列化。
            state['next'] += 1
            state['expected'].append(record['supervision']['target_output'])
            audited_records += 1
            if record['id'] not in indices:
                continue
            i, case = indices[record['id']], cases[cid]
            assert case['split'] == rows[i]['split'] and case['group_id'] == rows[i]['group_id']
            for arm in ARMS:
                for view in ['G', 'E_oracle', 'E_pred_42', 'E_pred_43', 'E_pred_44']:
                    evidence = None
                    if view == 'E_oracle' and not rows[i]['attack']:
                        continue
                    if view.startswith('E_pred_'):
                        prediction = predictions[int(view.rsplit('_', 1)[1])][cid]
                        if not prediction['detected']:
                            assert prediction['predicted_evidence'] == ''
                            continue
                        evidence = prediction['predicted_evidence']
                        message = case['context']['ordered_messages'][prediction['message_index']]
                        assert message['role'] == 'tool' and evidence == message['content'][prediction['start']:prediction['end']]
                    ids, audit = target_text_tokens(record, case, tokenizer, arm, 'G' if view == 'G' else 'E', evidence)
                    position, h = insert(ids)
                    maps[arm][view][i] = position
                    conflicts[f'{arm}/{view}'][h].add(rows[i]['y'])
                    lengths[f'{arm}/{view}'].append(audit['total_length'])
                    if audit['trimmed_tokens']:
                        trim_counts[f'{arm}/{view}'] += 1
                    log.write(json.dumps({'id': record['id'], 'view_name': view, 'token_hash': h, **audit}) + '\n')
    for arm in ARMS:
        assert (maps[arm]['G'] >= 0).all()
        assert [(i >= 0) for i in maps[arm]['E_oracle']] == [r['attack'] for r in rows]
        for seed in [42, 43, 44]:
            assert [(i >= 0) for i in maps[arm][f'E_pred_{seed}']] == [predictions[seed][r['case_id']]['detected'] for r in rows]
    for arm, views in maps.items():
        for view, mapping in views.items():
            np.save(out / f'{arm}_{view}_mapping.npy', mapping)
    with (out / 'manifest.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(out / 'unique_inputs.json', unique)
    audit = {'audited_records': audited_records, 'audited_cases': len(temporal),
             'initial_history_max_pairs': max_initial, 'temporal_boundary_verified': True,
             'eligible_rows': len(rows), 'unique_encoder_inputs': len(unique),
             'counts': dict(Counter(f'{r["split"]}/{r["y"]}' for r in rows)),
             'lengths': {key: {'n': len(v), 'max': max(v), 'mean': float(np.mean(v)), 'trimmed': trim_counts[key]} for key, v in lengths.items()},
             'conflicting_input_hashes': {k: sum(len(labels) > 1 for labels in values.values()) for k, values in conflicts.items()},
             'history_rule': 'initial snapshot targets + strictly previous blocks, tail256tokens; no draft/acceptance metadata',
             'S2_scope': 'current target available after generation, before release; not predictive detection'}
    write_json(out / 'data_audit.json', audit)
    print('主模型文本审计 ' + json.dumps(audit, ensure_ascii=False), flush=True)
    return rows, unique, sequences, maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--detector-dir', default='artifacts/modernbert_oracle_poc/predicted-evidence-20261105/detector')
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP'):
        raise SystemExit('需要受限wrapper')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path('artifacts/modernbert_oracle_poc/large-features-20261105-v2')
    detector = Path(args.detector_dir)
    model_info = json.loads(Path('artifacts/modernbert_oracle_poc/model-large.json').read_text())
    config = {'model': model_info, 'history_tokens': HISTORY_TOKENS, 'arms': list(ARMS),
              'backbone': 'frozen_eval_fp32_sdpa_reference_compile_false', 'pooling': 'last_hidden_nonpadding_mean_fp32',
              'serialization': 'v2-plus-target-tail-v1', 'protocol_sha256': file_sha256(out.parent / 'protocol.json'),
              'sources': {'records': file_sha256('artifacts/text_probe_v2/frozen/records.jsonl'),
                          'cases': file_sha256('artifacts/text_probe_v2/frozen/resolved_cases.jsonl'),
                          'original_manifest': file_sha256(source / 'manifest.jsonl')},
              'predictions_sha256': {str(seed): file_sha256(detector / f'seed-{seed}/predictions.jsonl') for seed in [42, 43, 44]},
              'code_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in
                              ['target_text_features.py', 'modernbert_data.py', 'modernbert_poc.py', 'modernbert_large_features.py']}}
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != config:
        raise ValueError('配置变化，不得混用缓存')
    write_json(out / 'config.json', config)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_info['local_path'], local_files_only=True)
    rows, unique, sequences, maps = build_target_inputs(out, tokenizer, source, detector)
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
                                      'unique_encoder_inputs': len(unique), 'hidden_size': cfg.hidden_size,
                                      'trainable_backbone_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    print(f'S1/S2特征提取完成：{out}', flush=True)


if __name__ == '__main__':
    main()
