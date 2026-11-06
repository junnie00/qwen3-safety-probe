import numpy as np
import pytest
import torch

from baseline.evidence_pipeline import encode_predicted, evidence_scores, predicted_input_ids


class Tokenizer:
    cls_token_id, sep_token_id = 1, 2

    def encode(self, text, add_special_tokens=False):
        ids = [ord(c) for c in text]
        return [1] + ids + [2] if add_special_tokens else ids

    def num_special_tokens_to_add(self, pair=False):
        return 2


def test_predicted_branch_reads_only_predicted_text_and_never_backfills():
    tokenizer = Tokenizer()
    assert predicted_input_ids('task', 'draft', {'detected': False, 'predicted_evidence': ''}, tokenizer) is None
    ids = predicted_input_ids('task', 'draft', {'detected': True, 'predicted_evidence': 'WRONG_SPAN'}, tokenizer)
    text = ''.join(map(chr, ids[1:-1]))
    assert 'WRONG_SPAN' in text and '[ATTACK]' in text
    with pytest.raises(ValueError):
        predicted_input_ids('task', 'draft', {'detected': False, 'predicted_evidence': 'gold'}, tokenizer)
    with pytest.raises(ValueError):
        predicted_input_ids('task', 'draft', {'detected': True, 'predicted_evidence': ''}, tokenizer)


def test_exact_input_cache_reuse_requires_matching_token_hash(tmp_path):
    source_rows = [{'E_hash': 'hash-a'}, {'E_hash': 'hash-b'}]
    source = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    unique = [{'id': 'pred-b', 'E_hash': 'hash-b'}, {'id': 'pred-a', 'E_hash': 'hash-a'}]
    vectors, info = encode_predicted(unique, [[1], [2]], source_rows, source, None, {}, tmp_path, {'test': True})
    assert info['new_encoder_inputs'] == 0 and info['exact_token_cache_hits'] == 2
    assert np.array_equal(vectors, source[[1, 0]])


def test_evidence_head_does_not_score_disabled_rows(tmp_path):
    head = torch.nn.Linear(2, 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1., 0.]]))
        head.bias.zero_()
    path = tmp_path / 'head.pt'
    torch.save({'hidden_size': 2, 'state_dict': head.state_dict()}, path)
    score = evidence_scores(path, np.array([[0., 1.], [1., 0.]], dtype=np.float32), np.array([1, -1, 0]))
    assert np.isnan(score[1])
    assert score[2] == .5
    assert np.isclose(score[0], torch.sigmoid(torch.tensor(1.)).item())
