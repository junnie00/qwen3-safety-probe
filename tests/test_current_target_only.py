import xml.etree.ElementTree as ET

import pytest

from baseline.current_target_only import current_only_tokens
from scripts.render_checked_trajectories import render


class Tokenizer:
    cls_token_id, sep_token_id = 1, 2

    def encode(self, text, add_special_tokens=False):
        ids = [ord(c) for c in text]
        return [1] + ids + [2] if add_special_tokens else ids

    def num_special_tokens_to_add(self, pair=False):
        return 2


class NoHistory(dict):
    def __getitem__(self, key):
        if key == 'history_pairs':
            raise AssertionError('S3不应读取模型历史')
        return super().__getitem__(key)


class OnlyCurrent(dict):
    def __getitem__(self, key):
        if key != 'target_output':
            raise AssertionError('只允许当前输出文本')
        return super().__getitem__(key)


def fixture():
    record = {'input': NoHistory({'current_draft': 'draft', 'context': {'ordered_messages': [
        {'role': 'system', 'content': 'policy'}, {'role': 'user', 'content': 'task'},
        {'role': 'assistant', 'content': 'visible chat'}, {'role': 'tool', 'content': 'material'}]}}),
        'supervision': OnlyCurrent({'target_output': 'CURRENT_OUTPUT', 'class_id': 0,
                                    'target_text_so_far': 'SECRET_FUTURE'})}
    return record, {'question': 'task', 'attack_str': 'ORACLE_ATTACK', 'reference_answer': 'SECRET_REFERENCE'}


def text(ids):
    return ''.join(map(chr, ids[1:-1]))


def test_only_current_text_is_added_and_no_history_access_occurs():
    record, case = fixture()
    ids, audit = current_only_tokens(record, case, Tokenizer(), 'G')
    assert '[TARGET_CURRENT]' in text(ids) and 'CURRENT_OUTPUT' in text(ids)
    assert '[TARGET_HISTORY]' not in text(ids) and 'SECRET' not in text(ids)
    assert 'visible chat' in text(ids)  # 不删除原本的聊天assistant消息。
    assert audit['history_field_included'] is False


def test_modifying_or_removing_history_never_changes_s3():
    record, case = fixture()
    record['input'] = dict(record['input'])
    before, _ = current_only_tokens(record, case, Tokenizer(), 'E')
    record['input']['history_pairs'] = [{'target': 'ARBITRARY_HISTORY', 'draft': 'OLD_DRAFT'}]
    after, _ = current_only_tokens(record, case, Tokenizer(), 'E')
    assert before == after
    record['supervision']['target_output'] = 'CHANGED_CURRENT'
    changed, _ = current_only_tokens(record, case, Tokenizer(), 'E')
    assert before != changed


def test_predicted_evidence_does_not_fall_back_to_oracle():
    record, case = fixture()
    ids, _ = current_only_tokens(record, case, Tokenizer(), 'E', evidence='PREDICTED_TEXT')
    assert 'PREDICTED_TEXT' in text(ids) and 'ORACLE_ATTACK' not in text(ids)


def test_current_text_preserved_under_message_trimming():
    record, case = fixture()
    record['input']['context']['ordered_messages'][-1]['content'] = 'x' * 5000
    ids, audit = current_only_tokens(record, case, Tokenizer(), 'G')
    assert len(ids) == 4096 and audit['trimmed_tokens'] > 0
    assert 'CURRENT_OUTPUT' in text(ids)
    record['supervision']['target_output'] = 'z' * 4096
    with pytest.raises(ValueError):
        current_only_tokens(record, case, Tokenizer(), 'E')


def test_empty_current_text_is_valid_without_substituting_history():
    record, case = fixture()
    record['supervision']['target_output'] = ''
    ids, audit = current_only_tokens(record, case, Tokenizer(), 'E')
    assert audit['current_target_tokens'] == 0
    assert text(ids).endswith('[TARGET_CURRENT]\n')


def test_checked_plot_escapes_less_than_in_axis_title(tmp_path):
    curve = [{'epoch': i, 'train_bce': .5, 'val_bce': .6, 'val_A_recall': .4, 'val_A_fpr': .02} for i in [0, 1]]
    path = tmp_path / 'curve.svg'
    render(curve, 1, 'A < B & C', path)
    ET.parse(path)
    assert '&lt;' in path.read_text() and '&amp;' in path.read_text()
