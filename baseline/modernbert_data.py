"""ModernBERT PoC 的白名单视图、分组隔离与长度审计；不改变原 Qwen pipeline。"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from baseline.text_data import class_id

SERIALIZATION_VERSION = 'modernbert-oracle-v2-cls-sep'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_lines(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def selected_messages(messages):
    indices = sorted(set(range(min(3, len(messages)))) |
                     set(range(max(0, len(messages) - 3), len(messages))))
    return [{'position': i, 'role': messages[i]['role'],
             'name': messages[i].get('name', ''), 'content': messages[i]['content']}
            for i in indices]


def view_fields(record, case):
    """禁止把 target、history_pairs、标签、ID 等旁路信息送入编码器。"""
    return {'task': case['question'], 'draft': record['input']['current_draft'],
            'messages': selected_messages(record['input']['context']['ordered_messages']),
            'attack': case.get('attack_str') or ''}


def tokenize_view(fields, view, tokenizer, max_length):
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix = enc('[TASK]\n' + fields['task'] + '\n[DRAFT]\n' + fields['draft'])
    special = tokenizer.num_special_tokens_to_add(pair=False)
    audit = {'view': view, 'messages': [], 'trimmed_tokens': 0}
    if view == 'E':
        content = prefix + enc('\n[ATTACK]\n' + fields['attack'])
        original = len(content) + special
        if original > max_length:
            raise ValueError(f'E必需字段超长：{original}>{max_length}')
    elif view == 'G':
        prefix += enc('\n[MESSAGES]\n')
        parts = [(enc('\n[ROLE=' + m['role'] +
                      (',NAME=' + m['name'] if m['name'] else '') + ']\n'),
                  enc(m['content']), m['position']) for m in fields['messages']]
        overhead = len(prefix) + special + sum(len(h) for h, _, _ in parts)
        if overhead > max_length:
            raise ValueError(f'G必需字段与角色头超长：{overhead}>{max_length}')
        n = sum(len(body) for _, body, _ in parts)
        budget = min(n, max_length - overhead)
        left, right = (budget + 1) // 2, budget // 2
        original = overhead + n
        content = list(prefix)
        offset = 0
        for header, body, position in parts:
            # 裁剪仅作用于消息content；所有消息角色头保留。
            keep = [(j, token) for j, token in enumerate(body)
                    if n <= budget or offset + j < left or offset + j >= n - right]
            content.extend(header)
            content.extend(token for _, token in keep)
            indices = [j for j, _ in keep]
            ranges = []
            for j in indices:
                if ranges and ranges[-1][1] == j:
                    ranges[-1][1] = j + 1
                else:
                    ranges.append([j, j + 1])
            audit['messages'].append({'position': position, 'original': len(body),
                                      'retained': len(keep), 'ranges': ranges})
            offset += len(body)
        audit['trimmed_tokens'] = n - budget
    else:
        raise ValueError(view)
    # 通用PreTrainedTokenizerFast的build_inputs_with_special_tokens可能是恒等映射。
    # 核验真正的backend单序列模板，再显式包裹；不硬编码模型token ID。
    boundary = tokenizer.encode('', add_special_tokens=True)
    expected = [tokenizer.cls_token_id, tokenizer.sep_token_id]
    if special != 2 or None in expected or boundary != expected:
        raise ValueError('不支持的单序列特殊token模板，需要重新适配tokenizer')
    tokens = [expected[0], *content, expected[1]]
    assert len(tokens) == len(content) + special
    assert len(tokens) <= max_length
    audit.update(original_length=original, retained_length=len(tokens))
    return tokens, audit


def build_dataset(records_path, cases_path, tokenizer, out_dir, max_length=4096):
    out_dir = Path(out_dir)
    cases = {}
    group_splits = {}
    for case in json_lines(cases_path):
        if case['id'] in cases:
            raise ValueError('重复case ID')
        cases[case['id']] = case
        group, split = case['group_id'], case['split']
        if group_splits.setdefault(group, split) != split:
            raise ValueError('来源组跨split泄漏')
    rows, tokens = [], {'G': [], 'E': []}
    seen, counts, excluded = set(), Counter(), Counter()
    raw_hashes = {'G': defaultdict(list), 'E': defaultdict(list)}
    token_hashes = {'G': defaultdict(list), 'E': defaultdict(list)}
    lengths, trimmed = defaultdict(list), Counter()
    total = 0
    with (out_dir / 'length_audit.jsonl').open('w') as audit_file:
        for record in json_lines(records_path):
            total += 1
            if record['id'] in seen:
                raise ValueError('重复record ID')
            seen.add(record['id'])
            case = cases[record['case_id']]
            for key in ['group_id', 'split']:
                if record[key] != case[key]:
                    raise ValueError(f'case关联不一致：{key}')
            supervision = record['supervision']
            y = supervision['class_id']
            if y != class_id(supervision['draft_hijacked'], supervision['target_hijacked']):
                raise ValueError('类别与行为标签不一致')
            if supervision['after_violation']:
                excluded['after_violation'] += 1
                continue
            if y is None:
                excluded['unknown_label'] += 1
                continue
            if y not in [0, 1, 2] or record['split'] not in ['train', 'val', 'test']:
                raise ValueError('非法标签/split')
            fields = view_fields(record, case)
            attack = bool(fields['attack'].strip())
            row = {'id': record['id'], 'case_id': case['id'], 'group_id': case['group_id'],
                   'split': case['split'], 'y': y, 'attack': attack,
                   'domain': case['domain'], 'variant': case['variant'],
                   'label_source': record.get('provenance', {}).get('label_source', 'unknown')}
            counts[f'{row["split"]}/{y}'] += 1
            for view in ['G', 'E']:
                if view == 'E' and not attack:
                    tokens[view].append(None)
                    continue
                ids, audit = tokenize_view(fields, view, tokenizer, max_length)
                token_hash = digest(ids)
                raw = {key: fields[key] for key in
                       (['task', 'draft', 'messages'] if view == 'G' else ['task', 'draft', 'attack'])}
                raw_hashes[view][digest(raw)].append(len(rows))
                token_hashes[view][token_hash].append(len(rows))
                row[view + '_hash'] = token_hash
                tokens[view].append(ids)
                key = f'{view}/{row["split"]}/{row["domain"]}/{y}'
                lengths[key].append(audit['original_length'])
                if audit['trimmed_tokens']:
                    trimmed[key] += 1
                audit_file.write(json.dumps({'id': row['id'], **audit}) + '\n')
            rows.append(row)
    collision_report = {}
    for kind, maps in [('raw', raw_hashes), ('tokens', token_hashes)]:
        for view, mapping in maps.items():
            conflicts = []
            cross_split = []
            for h, indices in mapping.items():
                if len({rows[i]['y'] for i in indices}) > 1:
                    conflicts.append({'hash': h, 'ids': [rows[i]['id'] for i in indices],
                                      'labels': [rows[i]['y'] for i in indices]})
                if len({rows[i]['split'] for i in indices}) > 1:
                    cross_split.append(h)
            collision_report[kind + '/' + view] = {
                'conflicting_inputs': len(conflicts),
                'affected_records': sum(len(c['ids']) for c in conflicts),
                'cross_split_identical_inputs': len(cross_split), 'conflicts': conflicts}
    summary = {'cases': len(cases), 'source_groups': len(group_splits), 'raw_records': total,
               'eligible_records': len(rows), 'counts': dict(counts), 'excluded': dict(excluded),
               'attack_train': dict(Counter(str(r['y']) for r in rows if r['split'] == 'train' and r['attack'])),
               'lengths': {key: {'n': len(v), 'min': min(v), 'max': max(v),
                                  'mean': sum(v) / len(v), 'trimmed': trimmed[key]}
                          for key, v in lengths.items()},
               'collisions': collision_report,
               'source_sha256': {'records': file_sha256(records_path), 'cases': file_sha256(cases_path)},
               'serialization': SERIALIZATION_VERSION, 'max_length': max_length}
    (out_dir / 'data_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (out_dir / 'manifest.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return rows, tokens, summary
