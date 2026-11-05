import numpy as np
import pytest
import torch

from baseline.modernbert_binary import (
    CONFIG, aggregate, binary_labels, binary_metrics, calibrate_binary,
    paired_bootstrap, predict_binary, train_binary_head,
)


def test_binary_label_conversion_and_unknown_rejection():
    assert np.array_equal(binary_labels([0, 1, 2]), [1, 0, 0])
    with pytest.raises(ValueError):
        binary_labels([0, None])


def test_binary_confusion_order_and_both_negatives():
    y = binary_labels([0, 0, 1, 2, 2])
    r = binary_metrics(y, np.array([1, 0, 1, 0, 0]))
    assert r['confusion'] == [[2, 1], [1, 1]]
    assert r['A_recall'] == .5 and r['A_fpr'] == 1 / 3
    assert r['A_precision'] == .5


def test_binary_calibration_float32_ties_can_predict_none():
    scores = np.array([.3, .3, .3], dtype=np.float32)
    threshold = calibrate_binary([1, 0, 0], scores, max_fpr=0)
    assert np.array_equal(predict_binary(scores, threshold), [0, 0, 0])


def test_binary_calibration_max_recall_then_lower_fpr():
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([.95, .75, .85, .2, .1])
    threshold = calibrate_binary(y, scores, max_fpr=0)
    assert threshold == .95
    assert binary_metrics(y, predict_binary(scores, threshold))['A_recall'] == .5
    threshold = calibrate_binary(y, scores, max_fpr=1 / 3)
    assert threshold == .75


def test_binary_aggregation_never_uses_disabled_nan_branch():
    g = np.array([.1, .2])
    aux = np.array([.8, np.nan])
    assert np.array_equal(aggregate(g, aux, np.array([True, False])), [.8, .2])


def test_paired_group_bootstrap_identical_predictions_have_zero_delta():
    y = np.array([1, 0, 1, 0])
    preds = {name: np.array([1, 0, 0, 0]) for name in ['M0', 'M1', 'M2', 'three_M2']}
    result = paired_bootstrap(y, preds, np.array(['a', 'a', 'b', 'b']), seed=42, iterations=20)
    assert result['groups'] == 2
    for interval in result['percentile_95'].values():
        assert interval == [0., 0.]


def test_training_scope_weights_and_checkpoint_replay(tmp_path):
    torch.set_num_threads(1)
    rows = []
    for split in ['train', 'val', 'test']:
        for attack in [False, True]:
            for label in [0, 1, 2]:
                rows.append({'split': split, 'attack': attack, 'y': label})
    features = np.random.default_rng(42).normal(size=(len(rows), 4)).astype('float32')
    config = {**CONFIG, 'max_epochs': 3, 'patience': 2, 'head_batch': 4}
    for name in ['G', 'G_aux', 'E']:
        x = features.copy()
        if name == 'E':
            x[[not r['attack'] for r in rows]] = np.nan
        scores, info = train_binary_head(name, x, rows, 42, tmp_path, config)
        assert info['pos_weight'] == 2
        assert info['train_n'] == (6 if name == 'G' else 3)
        assert info['val_n'] == (6 if name == 'G' else 3)
        ckpt = torch.load(tmp_path / f'{name}-seed-42.pt', weights_only=True)
        assert ckpt['label_mapping'] == {'A': 1, 'nonA': 0}
        assert ckpt['state_dict']['weight'].shape == (1, 4)
        assert ckpt['train_counts']['A'] == (2 if name == 'G' else 1)
        head = torch.nn.Linear(4, 1)
        head.load_state_dict(ckpt['state_dict'])
        valid = np.array([name != 'E' or r['attack'] for r in rows])
        with torch.inference_mode():
            replay = head(torch.tensor(x[valid])).squeeze(-1).sigmoid().numpy()
        assert np.allclose(scores[valid], replay)
        if name == 'E':
            assert np.isnan(scores[~valid]).all()
