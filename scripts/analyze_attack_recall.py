"""按既定large oracle M2结果分析A的召回/漏检，不据此回调模型或阈值。"""
import collections
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.modernbert_data import json_lines

TASK_FAMILIES = {'Task Automation', 'Business Intelligence', 'Conversational Agent', 'Research Assistance', 'Sentiment Analysis'}
TRANSFORM_FAMILIES = {'Substitution Ciphers', 'Base Encoding', 'Reverse Text', 'Emoji Substitution', 'Language Translation'}


def main():
    if not os.environ.get('POC_CGROUP'):
        raise SystemExit('需要资源限额wrapper')
    out = ROOT / 'artifacts/modernbert_oracle_poc/attack-recall-analysis-20261105'
    out.mkdir(parents=True, exist_ok=True)
    cases = {c['id']: c for c in json_lines(ROOT / 'artifacts/text_probe_v2/frozen/resolved_cases.jsonl')}
    attack_map = {}
    for split in ['train', 'test']:
        for family, attacks in json.loads((ROOT / f'data/bipia/text_attack_{split}.json').read_text()).items():
            for attack in attacks:
                attack_map[attack] = family
    rows = {r['id']: r for r in json_lines(ROOT / 'artifacts/modernbert_oracle_poc/run-20261104-v2/manifest.jsonl') if r['split'] == 'test' and r['y'] == 0}
    predictions = collections.defaultdict(dict)
    for p in json_lines(ROOT / 'artifacts/modernbert_oracle_poc/large-vs-base-20261105/test_predictions.jsonl'):
        if p['model'] == 'large' and p['configuration'] == 'M2' and p['id'] in rows:
            predictions[p['id']][p['seed']] = p['prediction_A']
    details = []
    for record in json_lines(ROOT / 'artifacts/text_probe_v2/frozen/records.jsonl'):
        if record['id'] not in rows:
            continue
        c = cases[record['case_id']]
        attack = c.get('attack_str') or ''
        if c['variant'].startswith('marker'):
            family = attack.replace(c['evaluator']['text'], '{marker}')
            if family.startswith('For downstream'):
                mechanism = 'marker/业务流程理由要求插入凭证'
            elif family.startswith('The actual request'):
                mechanism = 'marker/宣称任务已解决并替换回答'
            elif family.startswith('Processing note'):
                mechanism = 'marker/面向语言模型的处理指令'
            else:
                mechanism = 'marker/其他'
            kind = 'marker'
        else:
            family = attack_map.get(attack, 'UNMAPPED')
            mechanism = ('自然语言/改做无关任务' if family in TASK_FAMILIES else
                         '自然语言/改变输出表示或语言' if family in TRANSFORM_FAMILIES else
                         '自然语言/向答案夹带额外内容')
            kind = '自然语言'
        votes = predictions[record['id']]
        assert set(votes) == {42, 43, 44}
        count = sum(votes.values())
        details.append({'id': record['id'], 'case_id': c['id'], 'group_id': c['group_id'],
                        'domain': c['domain'], 'multi_turn': c['multi_turn'], 'variant': c['variant'],
                        'kind': kind, 'family': family, 'mechanism': mechanism,
                        'predictions': votes, 'hits_across_seeds': count,
                        'stability': '全部召回' if count == 3 else '全部漏检' if count == 0 else '种子间不一致',
                        'question': c['question'], 'attack': attack,
                        'current_draft': record['input']['current_draft'],
                        'target_output_for_audit_only': record['supervision']['target_output'],
                        'target_prefix_for_audit_only': record['supervision']['target_prefix_before'],
                        'history_length': len(record['input']['history_pairs'])})
    stats = {}
    for dimension in ['kind', 'mechanism', 'family', 'domain', 'multi_turn', 'variant']:
        groups = collections.defaultdict(list)
        for d in details:
            groups[str(d[dimension])].append(d)
        stats[dimension] = {}
        for name, members in groups.items():
            stats[dimension][name] = {'A_blocks': len(members), 'A_cases': len({d['case_id'] for d in members}),
                                     'source_groups': len({d['group_id'] for d in members}),
                                     'mean_recall': sum(d['hits_across_seeds'] for d in members) / (3 * len(members)),
                                     'seed_recall': {str(seed): sum(d['predictions'][seed] for d in members) / len(members) for seed in [42, 43, 44]},
                                     'stable_hits': sum(d['hits_across_seeds'] == 3 for d in members),
                                     'stable_misses': sum(d['hits_across_seeds'] == 0 for d in members),
                                     'mixed': sum(0 < d['hits_across_seeds'] < 3 for d in members)}
    result = {'scope': '当前large oracle M2，固定各seed验证阈值；仅测试真实A',
              'A_blocks': len(details), 'A_cases': len({d['case_id'] for d in details}),
              'source_groups': len({d['group_id'] for d in details}),
              'stability': dict(collections.Counter(d['stability'] for d in details)), 'statistics': stats,
              'limitations': ['攻击family是原BIPIA题材/目标分类，mechanism为人工规则归并，不是新训练标签。',
                              '全都漏检/召回表示跨三个seed的稳定性，不代表三个独立测试样本。',
                              '这是按A成功样本条件化的召回，不是各攻击的攻击成功率。',
                              '所有构造攻击均追加在tool材料末尾，本数据不覆盖任意位置、隐写或跨材料分布式攻击。',
                              '单轮/多轮与variant、攻击模板等混杂，不能据此作因果结论。']}
    (out / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with (out / 'A_cases.jsonl').open('w') as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    lines = ['# large oracle M2：A类召回与漏检的攻击模式分析', '',
             f'范围：{len(details)}个真实A块，来自{result["A_cases"]}个case、{result["source_groups"]}个来源组。',
             f'跨三seed稳定性：{result["stability"]}。', '',
             '平均召回为三个seed算术平均，不把同一块重复三次当独立样本。', '']
    for dim in ['kind', 'mechanism', 'family', 'domain', 'multi_turn']:
        lines += [f'## {dim}', '', '| 分组 | A块数 | A case数 | 来源组数 | 平均召回 | 全召回 | 全漏检 | 不一致 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name, s in sorted(stats[dim].items(), key=lambda item: (item[1]['mean_recall'], item[0])):
            safe_name = name.replace('|', '\\|').replace('\n', ' ')
            lines.append(f'| {safe_name} | {s["A_blocks"]} | {s["A_cases"]} | {s["source_groups"]} | {s["mean_recall"]:.2%} | {s["stable_hits"]} | {s["stable_misses"]} | {s["mixed"]} |')
        lines.append('')
    lines += ['## 代表性样本（确定性选择，不是随机总体样本）', '']
    for stability in ['全部漏检', '全部召回']:
        used = set()
        for d in details:
            if d['stability'] != stability or d['mechanism'] in used:
                continue
            used.add(d['mechanism'])
            lines += [f'### {stability}：{d["mechanism"]}', '', f'- ID：`{d["id"]}`',
                      f'- family：{d["family"]}；domain：{d["domain"]}；multi_turn：{d["multi_turn"]}',
                      '- 攻击原文（不可信引用）：', '```text', d['attack'], '```',
                      '- 当前草稿（输入的一部分）：', '```text', d['current_draft'], '```',
                      '- 主模型输出（仅用于事后审计，未送入Probe）：', '```text', d['target_output_for_audit_only'], '```', '']
    lines += ['## 限制', ''] + ['- ' + x for x in result['limitations']]
    lines += ['- 小组样本量不足时不作可靠性排名；完整逐块信息见A_cases.jsonl。',
              '- A块的主模型被带偏不要求当前草稿已显露攻击内容；Probe不读当前target，存在观察信息不足。', '']
    (out / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({k:result[k] for k in ['A_blocks','A_cases','source_groups','stability']},ensure_ascii=False))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
