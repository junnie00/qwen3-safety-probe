"""从可见tool材料检测并定位攻击段落；gold只用于监督和评估，不用于候选生成/推理。"""
import argparse
from collections import Counter, defaultdict
import gc
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from baseline.modernbert_data import digest, file_sha256, json_lines, selected_messages
from baseline.modernbert_poc import cache_features, write_json
from baseline.modernbert_large_features import benchmark_checked
from baseline.modernbert_trajectory import atomic_torch_save, fast_calibrate
from baseline.modernbert_binary import binary_metrics

DETECTOR_CONFIG = {'window_tokens': 192, 'stride_tokens': 96, 'max_length': 4096,
                   'epochs': 100, 'lr': .001, 'weight_decay': .0001, 'batch': 128,
                   'seeds': [42, 43, 44], 'clean_case_fpr_budget': .05,
                   'checkpoint_selection': 'minimum_unweighted_validation_case_BCE',
                   'serialization': 'task-tool-paragraph-v1', 'labels': {'no_attack': 0, 'attack': 1}}


def paragraphs(text):
    cursor = 0
    for separator in re.finditer(r'\r?\n[ \t]*\r?\n(?:[ \t]*\r?\n)*', text):
        segment = text[cursor:separator.start()]
        start = cursor + len(segment) - len(segment.lstrip())
        end = cursor + len(segment.rstrip())
        if start < end:
            yield start, end
        cursor = separator.end()
    segment = text[cursor:]
    start = cursor + len(segment) - len(segment.lstrip())
    end = cursor + len(segment.rstrip())
    if start < end:
        yield start, end


def candidate_inputs(case, tokenizer, config=DETECTOR_CONFIG):
    """仅question和可见messages决定候选；不读取attack_str/evaluator/行为标签。"""
    candidates, sequences = [], []
    for message in selected_messages(case['context']['ordered_messages']):
        if message['role'] != 'tool':
            continue
        text = message['content']
        for start, end in paragraphs(text):
            part = text[start:end]
            encoding = tokenizer(part, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoding['offset_mapping']
            spans = []
            if len(offsets) <= config['window_tokens']:
                spans = [(start, end)]
            else:
                for index in range(0, len(offsets), config['stride_tokens']):
                    last = min(index + config['window_tokens'], len(offsets))
                    spans.append((start + offsets[index][0], start + offsets[last - 1][1]))
                    if last == len(offsets):
                        break
            for left, right in spans:
                if left >= right:
                    continue
                raw = text[left:right]
                sequence = tokenizer.encode('[TASK]\n' + case['question'] + '\n[MATERIAL]\n' + raw, add_special_tokens=True)
                if len(sequence) > config['max_length']:
                    raise ValueError('任务加候选材料超长，不静默裁剪')
                assert sequence[0] == tokenizer.cls_token_id and sequence[-1] == tokenizer.sep_token_id
                candidates.append({'message_index': message['position'], 'start': left, 'end': right, 'text': raw,
                                   'G_hash': digest(sequence)})
                sequences.append(sequence)
    return candidates, sequences


def gold_span(case):
    attack = case.get('attack_str') or ''
    if not attack:
        return None
    hits = []
    for i, message in enumerate(case['context']['ordered_messages']):
        if message['role'] != 'tool':
            continue
        cursor = 0
        while True:
            start = message['content'].find(attack, cursor)
            if start < 0:
                break
            hits.append((i, start, start + len(attack)))
            cursor = start + len(attack)
    if len(hits) != 1:
        raise ValueError(f'需要唯一可审计的注入span，实际{len(hits)}：{case["id"]}')
    return hits[0]


def intersection(candidate, gold):
    if gold is None or candidate['message_index'] != gold[0]:
        return 0
    return max(0, min(candidate['end'], gold[2]) - max(candidate['start'], gold[1]))


def build_candidates(cases, tokenizer, out):
    rows, sequences = [], []
    counts, keys = Counter(), defaultdict(set)
    group_splits = {}
    for case in cases:
        if group_splits.setdefault(case['group_id'], case['split']) != case['split']:
            raise ValueError('来源组跨split')
        candidates, seqs = candidate_inputs(case, tokenizer)
        if not candidates:
            raise ValueError('无可见候选材料，需要显式处理空上下文')
        gold = gold_span(case)
        for index, (candidate, seq) in enumerate(zip(candidates, seqs)):
            label = int(intersection(candidate, gold) > 0)
            row = {'id': f'{case["id"]}/candidate-{index}', 'case_id': case['id'], 'split': case['split'],
                   'group_id': case['group_id'], 'domain': case['domain'], 'attack_label': label, **candidate}
            rows.append(row)
            sequences.append(seq)
            counts[f'{case["split"]}/{label}'] += 1
            keys[candidate['G_hash']].add((case['split'], label))
        if gold and not any(intersection(c, gold) for c in candidates):
            raise ValueError('候选窗口未覆盖gold，停止而非排除样本')
    audit = {'case_counts': dict(Counter(c['split'] for c in cases)), 'candidate_counts': dict(counts),
             'candidates': len(rows), 'unique_inputs': len(keys),
             'cross_split_identical_inputs': sum(len({s for s, y in v}) > 1 for v in keys.values()),
             'conflicting_labels_for_identical_inputs': sum(len({y for s, y in v}) > 1 for v in keys.values()),
             'label_scope': '构造攻击span重叠，不是主模型是否中招',
             'limitations': '所有攻击在tool材料末尾；段落边界定位可能较容易，不代表任意位置注入'}
    write_json(out / 'candidate_audit.json', audit)
    with (out / 'candidates.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('候选审计 ' + json.dumps(audit, ensure_ascii=False), flush=True)
    return rows, sequences, audit


def case_indices(cases, rows):
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row['case_id']].append(index)
    return [np.array(grouped[c['id']], dtype=int) for c in cases]


def select_predictions(public_cases, candidates, scores, threshold):
    """不接收gold或行为标签；输出真实材料中的最高分原文窗口。"""
    indexes = case_indices(public_cases, candidates)
    output = []
    for case, group in zip(public_cases, indexes):
        winner = int(group[np.argmax(scores[group])])
        candidate = candidates[winner]
        score = float(scores[winner])
        detected = score >= threshold
        output.append({'case_id': case['id'], 'score_attack': score, 'detected': bool(detected),
                       'candidate_id': candidate['id'], 'message_index': candidate['message_index'],
                       'start': candidate['start'], 'end': candidate['end'],
                       'predicted_evidence': candidate['text'] if detected else ''})
    return output


def detector_metrics(cases, predictions):
    groups = {}
    for split in ['train', 'val', 'test']:
        subset = [(c, p) for c, p in zip(cases, predictions) if c['split'] == split]
        y = np.array([int(bool(c.get('attack_str'))) for c, p in subset])
        pred = np.array([int(p['detected']) for c, p in subset])
        m = binary_metrics(y, pred)
        coverages, precisions = [], []
        exact = 0
        for case, prediction in subset:
            gold = gold_span(case)
            if gold is None:
                continue
            overlap = intersection(prediction, gold) if prediction['detected'] else 0
            coverages.append(overlap / (gold[2] - gold[1]))
            precisions.append(overlap / (prediction['end'] - prediction['start']) if prediction['detected'] else 0.)
            exact += int(prediction['predicted_evidence'] == case['attack_str'])
        groups[split] = {'cases': len(subset), 'attack_cases': int(y.sum()), 'clean_cases': int((1 - y).sum()),
                         'attack_recall': m['A_recall'], 'clean_fpr': m['A_fpr'], 'attack_precision': m['A_precision'],
                         'confusion_no_attack_then_attack': m['confusion'],
                         'mean_gold_character_coverage': float(np.mean(coverages)),
                         'mean_evidence_character_precision_on_attack_cases': float(np.mean(precisions)),
                         'full_gold_coverage_rate': float(np.mean(np.array(coverages) == 1)),
                         'exact_evidence_rate': exact / len(coverages)}
    return groups


def train_detector(features, rows, cases, seed, out, config):
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    val_windows = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    case_groups = case_indices(cases, rows)
    val_cases = np.array([i for i, c in enumerate(cases) if c['split'] == 'val'])
    y = np.array([r['attack_label'] for r in rows], dtype=np.float32)
    case_y = np.array([bool(c.get('attack_str')) for c in cases], dtype=np.float32)
    xt = torch.tensor(np.asarray(features[train]).copy())
    yt = torch.tensor(y[train])
    xv = torch.tensor(np.asarray(features[val_windows]).copy())
    yv = torch.tensor(y[val_windows])
    relative = {int(index): position for position, index in enumerate(val_windows)}
    val_case_windows = [np.array([relative[int(i)] for i in case_groups[k]]) for k in val_cases]
    weight = (len(train) - float(yt.sum())) / float(yt.sum())
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = torch.nn.Linear(xt.shape[1], 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    best, best_epoch, curve = float('inf'), None, []
    start = time.perf_counter()
    with (out / 'trajectory.jsonl').open('w') as log:
        for epoch in range(1, config['epochs'] + 1):
            head.train()
            total = 0.
            for batch in torch.randperm(len(train), generator=generator).split(config['batch']):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(head(xt[batch]).squeeze(-1), yt[batch])
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            head.eval()
            with torch.inference_mode():
                logits = head(xv).squeeze(-1)
                maxima = torch.stack([logits[indices].max() for indices in val_case_windows])
                case_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(maxima, torch.tensor(case_y[val_cases])))
                window_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, yv))
                probabilities = maxima.sigmoid().numpy()
            threshold = fast_calibrate(case_y[val_cases].astype(int), probabilities, config['clean_case_fpr_budget'])
            metrics = binary_metrics(case_y[val_cases].astype(int), (probabilities.astype(np.float64) >= threshold).astype(int))
            record = {'epoch': epoch, 'train_window_weighted_bce': total / len(train), 'val_window_bce': window_loss,
                      'val_case_bce': case_loss, 'val_attack_recall': metrics['A_recall'], 'val_clean_fpr': metrics['A_fpr']}
            curve.append(record)
            checkpoint = {'state_dict': head.state_dict(), 'optimizer_state': optimizer.state_dict(),
                          'torch_rng_state': torch.get_rng_state(), 'shuffle_rng_state': generator.get_state(),
                          'hidden_size': xt.shape[1], 'seed': seed, 'epoch': epoch, 'config': config,
                          'label_mapping': {'no_attack': 0, 'attack': 1}, 'pos_weight': weight, 'metrics': record}
            atomic_torch_save(checkpoint, checkpoints / f'epoch-{epoch:03d}.pt')
            if case_loss < best:
                best, best_epoch = case_loss, epoch
                atomic_torch_save(checkpoint, out / 'best.pt')
            if epoch == config['epochs']:
                atomic_torch_save(checkpoint, out / 'final.pt')
            log.write(json.dumps(record) + '\n')
            log.flush()
        best_ckpt = torch.load(out / 'best.pt', weights_only=True)
        head.load_state_dict(best_ckpt['state_dict'])
        all_scores = np.empty(len(rows), dtype=np.float32)
        with torch.inference_mode():
            for offset in range(0, len(rows), 256):
                all_scores[offset:offset + 256] = head(torch.tensor(np.asarray(features[offset:offset + 256]).copy())).squeeze(-1).sigmoid().numpy()
    val_scores = np.array([float(all_scores[case_groups[i]].max()) for i in val_cases])
    threshold = fast_calibrate(case_y[val_cases].astype(int), val_scores, config['clean_case_fpr_budget'])
    public_cases = [{'id': c['id']} for c in cases]
    predictions = select_predictions(public_cases, rows, all_scores, threshold)
    np.save(out / 'candidate_scores.npy', all_scores)
    with (out / 'predictions.jsonl').open('w') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + '\n')
    result = {'seed': seed, 'best_epoch': best_epoch, 'threshold': threshold, 'pos_weight': weight,
              'elapsed_seconds': time.perf_counter() - start, 'metrics': detector_metrics(cases, predictions)}
    write_json(out / 'results.json', result)
    print('证据检测器 ' + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    if not os.environ.get('POC_CGROUP'):
        raise SystemExit('需要资源限额wrapper')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        raise ValueError('使用新目录，不覆盖检测实验')
    model_info = json.loads(Path('artifacts/modernbert_oracle_poc/model.json').read_text())
    cases = list(json_lines('artifacts/text_probe_v2/frozen/resolved_cases.jsonl'))
    config = {'detector': DETECTOR_CONFIG, 'model': model_info,
              'source_sha256': file_sha256('artifacts/text_probe_v2/frozen/resolved_cases.jsonl'),
              'code_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in
                              ['evidence_detector.py', 'modernbert_data.py', 'modernbert_large_features.py', 'modernbert_poc.py']}}
    write_json(out / 'config.json', config)
    tokenizer = AutoTokenizer.from_pretrained(model_info['local_path'], local_files_only=True)
    rows, sequences, audit = build_candidates(cases, tokenizer, out)
    torch.cuda.set_per_process_memory_fraction(.5)
    model_cfg = AutoConfig.from_pretrained(model_info['local_path'], local_files_only=True)
    model_cfg.reference_compile = False
    model = AutoModel.from_pretrained(model_info['local_path'], config=model_cfg, local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    tokens = {'G': sequences, 'E': []}
    batches = benchmark_checked(model, tokenizer, tokens, out)
    contract = {'config': config, 'candidate_hash': digest(rows)}
    write_json(out / 'cache_contract.json', contract)
    features = cache_features(model, tokenizer, rows, tokens, 'G', out, contract, batches)
    del model, tokens, sequences
    gc.collect()
    torch.cuda.empty_cache()
    results = {str(seed): train_detector(features, rows, cases, seed, out / f'seed-{seed}', DETECTOR_CONFIG)
               for seed in DETECTOR_CONFIG['seeds']}
    write_json(out / 'results.json', results)
    write_json(out / 'completed.json', {'config_hash': digest(config), 'elapsed_seconds': time.perf_counter() - started})
    print(f'证据检测与定位完成：{out}', flush=True)


if __name__ == '__main__':
    main()
