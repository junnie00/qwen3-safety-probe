import copy

import pytest

from baseline.target_text_features import target_text_tokens


class Tokenizer:
    cls_token_id, sep_token_id = 1, 2

    def encode(self, text, add_special_tokens=False):
        ids = [ord(c) for c in text]
        return [1] + ids + [2] if add_special_tokens else ids

    def num_special_tokens_to_add(self, pair=False):
        return 2


class ForbiddenSupervision(dict):
    def __getitem__(self, key):
        raise AssertionError('S1不应读取supervision')


def inputs():
    context = {'ordered_messages': [{'role': 'system', 'content': 'policy'},
                                    {'role': 'user', 'content': 'task'},
                                    {'role': 'tool', 'content': 'material'}]}
    record = {'input': {'context': context, 'current_draft': 'draft',
                         'history_pairs': [{'target': 'past ', 'draft': 'SECRET_PAST_DRAFT',
                                            'accepted_draft_tokens': 9}, {'target': 'answer'}]},
              'supervision': {'target_output': 'CURRENT_TARGET', 'target_text_so_far': 'SECRET_FUTURE',
                              'class_id': 0, 'task_label': 1}}
    case = {'question': 'task', 'attack_str': 'ORACLE_ATTACK', 'reference_answer': 'SECRET_REFERENCE'}
    return record, case


def readable(ids):
    return ''.join(map(chr, ids[1:-1]))


def test_s1_never_reads_current_supervision_and_includes_only_past_targets():
    record, case = inputs()
    record['supervision'] = ForbiddenSupervision()
    ids, audit = target_text_tokens(record, case, Tokenizer(), 'S1', 'G')
    text = readable(ids)
    assert 'past answer' in text
    assert 'CURRENT_TARGET' not in text and 'SECRET' not in text
    assert audit['current_target_tokens'] == 0


def test_s1_unchanged_by_current_target_or_history_draft_metadata():
    record, case = inputs()
    before, _ = target_text_tokens(record, case, Tokenizer(), 'S1', 'E')
    record['supervision']['target_output'] = 'DIFFERENT_CURRENT'
    record['input']['history_pairs'][0]['draft'] = 'DIFFERENT_DRAFT'
    record['input']['history_pairs'][0]['accepted_draft_tokens'] = 999
    after, _ = target_text_tokens(record, case, Tokenizer(), 'S1', 'E')
    assert before == after


def test_s2_current_output_only_not_labels_or_future():
    record, case = inputs()
    ids, audit = target_text_tokens(record, case, Tokenizer(), 'S2', 'G')
    text = readable(ids)
    assert '[TARGET_CURRENT]' in text and 'CURRENT_TARGET' in text and 'past answer' in text
    assert 'SECRET' not in text and 'class_id' not in text
    assert audit['current_target_tokens'] == len('CURRENT_TARGET')


def test_history_tail_is_exactly_last_256_tokens():
    record, case = inputs()
    record['input']['history_pairs'] = [{'target': 'DROP_PREFIX' + 'x' * 256}]
    ids, audit = target_text_tokens(record, case, Tokenizer(), 'S1', 'E')
    assert 'DROP_PREFIX' not in readable(ids)
    assert audit['history_retained_tokens'] == 256
    assert readable(ids).endswith('x' * 256)


def test_predicted_evidence_never_falls_back_to_gold():
    record, case = inputs()
    ids, _ = target_text_tokens(record, case, Tokenizer(), 'S2', 'E', evidence='PREDICTED_WRONG_SPAN')
    assert 'PREDICTED_WRONG_SPAN' in readable(ids) and 'ORACLE_ATTACK' not in readable(ids)


def test_length_budget_preserves_main_text_and_trims_only_material():
    record, case = inputs()
    record['input']['context']['ordered_messages'][-1]['content'] = 'z' * 5000
    ids, audit = target_text_tokens(record, case, Tokenizer(), 'S2', 'G')
    assert len(ids) == 4096 and audit['trimmed_tokens'] > 0
    assert 'CURRENT_TARGET' in readable(ids) and 'past answer' in readable(ids)
    record['supervision']['target_output'] = 'x' * 4096
    with pytest.raises(ValueError):
        target_text_tokens(record, case, Tokenizer(), 'S2', 'E')


def test_materialized_mapping_preserves_disabled_rows_and_order():
    import numpy as np
    from baseline.target_text_ablation import materialize
    vectors = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    result = materialize(vectors, np.array([1, -1, 0]))
    assert np.array_equal(result[[0, 2]], vectors[[1, 0]])
    assert np.isnan(result[1]).all()


def test_temporal_arm_bootstrap_pairs_same_source_groups():
    import numpy as np
    from baseline.target_text_ablation import paired_intervals
    y = np.array([1, 0, 1, 0])
    pred = {arm: np.array([1, 0, 0, 0]) for arm in ['S0', 'S1', 'S2']}
    result = paired_intervals(y, pred, np.array(['g1', 'g1', 'g2', 'g2']), 42, 20)
    assert result['groups'] == 2
    assert all(interval == [0., 0.] for interval in result['percentile_95'].values())
