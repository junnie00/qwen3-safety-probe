import copy
import json

import numpy as np
import torch

from baseline.evidence_detector import (
    DETECTOR_CONFIG, candidate_inputs, gold_span, intersection, paragraphs,
    select_predictions, train_detector,
)


class CharacterTokenizer:
    cls_token_id, sep_token_id = 1, 2

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i + 1) for i in range(len(text))]}

    def encode(self, text, add_special_tokens=True):
        ids = [ord(c) for c in text]
        return [1] + ids + [2] if add_special_tokens else ids


def example_case():
    return {'id': 'x', 'question': 'summarize', 'attack_str': 'ignore task',
            'context': {'ordered_messages': [
                {'role': 'system', 'content': 'trusted instructions'},
                {'role': 'user', 'content': 'summarize'},
                {'role': 'tool', 'content': 'normal text\n\nignore task'}]}}


def test_paragraph_positions_and_candidates_do_not_depend_on_gold():
    case = example_case()
    a, seq_a = candidate_inputs(case, CharacterTokenizer())
    without_gold = copy.deepcopy(case)
    without_gold.pop('attack_str')
    without_gold['evaluator'] = {'kind': 'SECRET'}
    b, seq_b = candidate_inputs(without_gold, CharacterTokenizer())
    assert a == b and seq_a == seq_b
    assert [c['text'] for c in a] == ['normal text', 'ignore task']
    assert all(c['message_index'] == 2 for c in a)
    gold = gold_span(case)
    assert gold == (2, 13, 24)
    assert intersection(a[0], gold) == 0 and intersection(a[1], gold) == 11
    assert list(paragraphs('  a\r\n\r\n b  ')) == [(2, 3), (8, 9)]


def test_long_windows_use_text_offsets_without_gold_boundaries():
    case = example_case()
    case['context']['ordered_messages'][2]['content'] = 'abcdefghijklmnopqrst'
    config = {**DETECTOR_CONFIG, 'window_tokens': 8, 'stride_tokens': 4}
    candidates, _ = candidate_inputs(case, CharacterTokenizer(), config)
    assert [(x['start'], x['end']) for x in candidates] == [(0, 8), (4, 12), (8, 16), (12, 20)]
    assert all(x['text'] == 'abcdefghijklmnopqrst'[x['start']:x['end']] for x in candidates)


def test_inference_outputs_wrong_span_if_classifier_wrong_never_gold_fills():
    candidates = [dict(id='a/0', case_id='a', text='normal', message_index=2, start=0, end=6),
                  dict(id='a/1', case_id='a', text='attack', message_index=2, start=8, end=14)]
    scores = np.array([.9, .2])
    result = select_predictions([{'id': 'a'}], candidates, scores, .5)[0]
    assert result['predicted_evidence'] == 'normal'
    hidden = select_predictions([{'id': 'a'}], candidates, scores, .95)[0]
    assert not hidden['detected'] and hidden['predicted_evidence'] == ''


def test_small_detector_training_preserves_semantics_and_checkpoints(tmp_path):
    torch.set_num_threads(1)
    cases, rows = [], []
    for split in ['train', 'val', 'test']:
        for attacked in [False, True]:
            case_id = f'{split}-{attacked}'
            cases.append({'id': case_id, 'split': split, 'attack_str': 'attack' if attacked else None,
                          'context': {'ordered_messages': [{'role': 'tool', 'content': 'normal\n\nattack'}]}})
            for j, text in enumerate(['normal', 'attack']):
                rows.append({'id': f'{case_id}/{j}', 'case_id': case_id, 'split': split,
                             'attack_label': int(attacked and j == 1), 'message_index': 0,
                             'text': text, 'start': j * 8, 'end': j * 8 + 6})
    features = np.random.default_rng(42).normal(size=(len(rows), 4)).astype('float32')
    config = {**DETECTOR_CONFIG, 'epochs': 2, 'batch': 4}
    result = train_detector(features, rows, cases, 42, tmp_path, config)
    assert result['metrics']['test']['attack_cases'] == 1
    assert result['metrics']['val']['clean_fpr'] <= .05
    checkpoint = torch.load(tmp_path / 'best.pt', weights_only=True)
    assert checkpoint['label_mapping'] == {'no_attack': 0, 'attack': 1}
    assert len((tmp_path / 'trajectory.jsonl').read_text().splitlines()) == 2
    assert len(list((tmp_path / 'checkpoints').glob('*.pt'))) == 2
    assert len([json.loads(x) for x in (tmp_path / 'predictions.jsonl').read_text().splitlines()]) == 6
