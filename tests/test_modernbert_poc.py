import json

import numpy as np
import pytest
import torch

from baseline.modernbert_data import build_dataset, selected_messages, tokenize_view, view_fields
from baseline.modernbert_heads import calibrate, combined_scores, metrics, predict, run_heads
from baseline.modernbert_poc import mean_pool


class CharTokenizer:
    cls_token_id = 1
    sep_token_id = 2

    def encode(self, text, add_special_tokens=False):
        ids = [ord(c) for c in text]
        return [1] + ids + [2] if add_special_tokens else ids

    def num_special_tokens_to_add(self, pair=False):
        return 2


def test_generic_fast_tokenizer_uses_backend_special_template():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.WordLevel({'[UNK]': 0, '[CLS]': 1, '[SEP]': 2, 'task': 3}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 1), ('[SEP]', 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                       cls_token='[CLS]', sep_token='[SEP]')
    assert tokenizer.build_inputs_with_special_tokens([3]) == [3]
    fields = {'task': 'task', 'draft': 'task', 'attack': 'task', 'messages': []}
    ids, audit = tokenize_view(fields, 'E', tokenizer, 100)
    assert ids[0] == tokenizer.cls_token_id and ids[-1] == tokenizer.sep_token_id
    assert len(ids) == audit['original_length'] == audit['retained_length']


def test_message_selection_deduplicates_positions_not_content():
    messages = [{'role': 'user', 'content': 'same'} for _ in range(8)]
    assert [m['position'] for m in selected_messages(messages)] == [0, 1, 2, 5, 6, 7]
    assert len(selected_messages(messages[:3])) == 3


def test_input_whitelist_no_target_history_or_labels():
    record = {'input': {'current_draft': 'draft', 'context': {'ordered_messages': [
        {'role': 'tool', 'content': 'context'}]}, 'history_pairs': [{'target': 'SECRET_HISTORY'}]},
        'supervision': {'target_output': 'SECRET_TARGET', 'class_id': 0}}
    case = {'question': 'task', 'attack_str': 'attack', 'reference_answer': 'SECRET_REFERENCE', 'id': 'SECRET_ID'}
    fields = view_fields(record, case)
    text = json.dumps(fields)
    assert 'SECRET' not in text
    assert fields['attack'] == 'attack'


def test_truncate_only_message_content_keep_task_draft_roles():
    fields = {'task': 'TASK_VALUE', 'draft': 'DRAFT_VALUE', 'attack': 'ATTACK_VALUE',
              'messages': [{'position': 0, 'role': 'tool', 'name': 'email', 'content': 'a' * 200}]}
    ids, audit = tokenize_view(fields, 'G', CharTokenizer(), 100)
    text = ''.join(map(chr, ids[1:-1]))
    assert 'TASK_VALUE' in text and 'DRAFT_VALUE' in text and 'ROLE=tool,NAME=email' in text
    assert len(ids) == 100 and audit['trimmed_tokens'] > 0
    assert len(audit['messages'][0]['ranges']) == 2
    e, ea = tokenize_view(fields, 'E', CharTokenizer(), 100)
    assert 'ATTACK_VALUE' in ''.join(map(chr, e)) and ea['trimmed_tokens'] == 0
    with pytest.raises(ValueError, match='必需字段'):
        tokenize_view({**fields, 'attack': 'x' * 200}, 'E', CharTokenizer(), 100)


def test_mean_pool_ignores_padding():
    hidden = torch.tensor([[[1., 3.], [3., 5.], [100., 100.]]])
    assert torch.equal(mean_pool(hidden, torch.tensor([[1, 1, 0]])), torch.tensor([[2., 4.]]))


def test_disabled_branch_never_votes_and_a_has_priority():
    g = np.array([[.1, .7, .2], [.1, .2, .7]])
    e = np.array([[.8, .1, .1], [np.nan, np.nan, np.nan]])
    scores = combined_scores(g, e, np.array([True, False]))
    assert np.array_equal(predict(scores, .5, .5), [0, 2])
    assert np.array_equal(scores[1], g[1])


def test_calibrate_ties_can_predict_no_a_in_float32():
    y = np.array([0, 1, 2])
    scores = np.full((3, 3), 1 / 3, dtype=np.float32)
    a, b = calibrate(y, scores, max_fpr=0)
    pred = predict(scores, a, b)
    assert not np.any(pred == 0)
    assert metrics(y, pred)['A_fpr'] == 0


def test_calibration_respects_budget_and_keeps_b_c_as_negatives():
    y = np.array([0, 0, 1, 2, 2])
    scores = np.array([[.9, .05, .05], [.8, .1, .1], [.85, .1, .05], [.2, .2, .6], [.1, .2, .7]])
    a, b = calibrate(y, scores, max_fpr=0)
    result = metrics(y, predict(scores, a, b))
    assert result['A_recall'] == .5
    assert result['A_fpr'] == 0


def test_audit_preserves_attack_c_and_reports_conflicting_inputs(tmp_path):
    case = {'id': 'c', 'group_id': 'g', 'split': 'train', 'domain': 'email', 'variant': 'marker',
            'question': 'task', 'attack_str': 'attack'}
    record = {'id': 'r0', 'case_id': 'c', 'group_id': 'g', 'split': 'train',
              'input': {'context': {'ordered_messages': [{'role': 'tool', 'content': 'context'}]},
                        'current_draft': 'same'},
              'supervision': {'class_id': 2, 'draft_hijacked': 0, 'target_hijacked': 0, 'after_violation': False}}
    second = {**record, 'id': 'r1', 'supervision': {**record['supervision'], 'class_id': 0, 'target_hijacked': 1}}
    (tmp_path / 'cases').write_text(json.dumps(case) + '\n')
    (tmp_path / 'records').write_text('\n'.join(map(json.dumps, [record, second])))
    rows, tokens, audit = build_dataset(tmp_path / 'records', tmp_path / 'cases', CharTokenizer(), tmp_path)
    assert len(rows) == 2 and all(r['attack'] for r in rows)
    assert audit['attack_train'] == {'2': 1, '0': 1}
    assert audit['collisions']['tokens/G']['conflicting_inputs'] == 1
    assert tokens['G'][0] == tokens['G'][1]
    (tmp_path / 'cases').write_text('\n'.join(map(json.dumps, [case, {**case, 'id': 'c2', 'split': 'test'}])))
    with pytest.raises(ValueError, match='跨split'):
        build_dataset(tmp_path / 'records', tmp_path / 'cases', CharTokenizer(), tmp_path)


def test_tiny_training_calibration_and_checkpoint_contract(tmp_path):
    torch.set_num_threads(1)
    rows = []
    for split in ['train', 'val', 'test']:
        for attack in [False, True]:
            for y in range(3):
                rows.append({'id': f'{split}-{attack}-{y}', 'split': split, 'attack': attack,
                             'y': y, 'domain': 'email', 'variant': 'synthetic',
                             'label_source': 'synthetic', 'group_id': f'{split}-{y}'})
    rng = np.random.default_rng(42)
    g = rng.normal(size=(len(rows), 4)).astype('float32')
    e = g.copy()
    e[[not r['attack'] for r in rows]] = np.nan
    cfg = {'lr': 1e-3, 'weight_decay': 1e-4, 'head_batch': 8, 'max_epochs': 2,
           'patience': 1, 'seeds': [42], 'max_a_fpr': .05, 'bootstrap_iterations': 10}
    result = run_heads(g, e, rows, tmp_path, cfg)
    assert set(result['seeds']['42']) >= {'M0', 'M1', 'M2', 'bootstrap'}
    checkpoint = torch.load(tmp_path / 'E-seed-42.pt', weights_only=True)
    assert checkpoint['classes'] == ['A', 'B', 'C']
    assert checkpoint['train_class_counts'] == [1, 1, 1]
    for model in ['M0', 'M1', 'M2']:
        assert result['seeds']['42'][model]['validation']['A_fpr'] <= .05
