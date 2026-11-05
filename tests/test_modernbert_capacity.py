import json

import numpy as np
import torch

from baseline.modernbert_binary import calibrate_binary
from baseline.modernbert_capacity import TRAIN_CONFIG, capacity_bootstrap
from baseline.modernbert_trajectory import fast_calibrate, train_trajectory


def test_sorted_calibration_exactly_matches_exhaustive_with_ties():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y = np.r_[np.ones(8, dtype=int), np.zeros(52, dtype=int)]
        scores = np.round(rng.random(len(y)), 2).astype('float32')
        for budget in [0, .05, .2, 1.]:
            assert fast_calibrate(y, scores, budget) == calibrate_binary(y, scores, budget)
    assert fast_calibrate([1, 0], np.array([.5, .5], dtype=np.float32), 0) > .5


def test_full_trajectory_does_not_early_stop_and_saves_replayable_epochs(tmp_path):
    torch.set_num_threads(1)
    rows = [{'id': f'{split}-{y}', 'group_id': split, 'split': split, 'attack': True, 'y': y}
            for split in ['train', 'val', 'test'] for y in [0, 1, 2]]
    features = np.random.default_rng(42).normal(size=(len(rows), 4)).astype('float32')
    cfg = {**TRAIN_CONFIG, 'epochs': 3, 'diagnostic_patience': 1, 'lr': 0.0}
    summary = train_trajectory('G', features, rows, 42, tmp_path, cfg, {'test': True})
    assert summary['epochs_completed'] == 3
    assert summary['best_epoch'] == 1
    assert summary['simulated_patience10_stop_epoch'] == 2
    curve = [json.loads(line) for line in (tmp_path / 'trajectory.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in curve] == [0, 1, 2, 3]
    assert all(r['val_A_fpr'] <= .05 for r in curve)
    scores = np.load(tmp_path / 'validation_scores.npy')
    assert scores.shape == (4, 3) and np.isfinite(scores).all()
    assert len(list((tmp_path / 'checkpoints').glob('*.pt'))) == 4
    final = torch.load(tmp_path / 'final.pt', weights_only=True)
    best = torch.load(tmp_path / 'best.pt', weights_only=True)
    assert final['epoch'] == 3 and best['epoch'] == 1
    assert {'optimizer_state', 'torch_rng_state', 'shuffle_rng_state', 'feature_identity'} <= final.keys()
    assert (tmp_path / 'trajectory.svg').read_text().startswith('<svg')
    assert (tmp_path / 'trajectory.csv').exists()
    head = torch.nn.Linear(4, 1)
    head.load_state_dict(final['state_dict'])
    with torch.inference_mode():
        actual = head(torch.tensor(features[3:6])).squeeze(-1).sigmoid().numpy()
    assert np.allclose(actual, scores[3])


def test_capacity_bootstrap_is_paired_by_source_group():
    y = np.array([1, 0, 1, 0])
    predictions = {f'{model}_{name}': np.array([1, 0, 0, 0])
                   for model in ['base', 'large'] for name in ['M0', 'M1', 'M2']}
    result = capacity_bootstrap(y, predictions, np.array(['a', 'a', 'b', 'b']), 42, iterations=20)
    assert result['groups'] == 2
    assert all(v == [0., 0.] for v in result['percentile_95'].values())


def test_large_benchmark_rejects_inconsistent_batch_instead_of_relaxing_tolerance(monkeypatch, tmp_path):
    import baseline.modernbert_large_features as module
    monkeypatch.setattr(module, 'memory_pressure', lambda: (False, 100, 200))
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (20 * 1024**3, 24 * 1024**3))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    def encode(model, tokenizer, sequences, device):
        return np.ones((len(sequences), 3), dtype='float32') + (.1 if len(sequences) == 4 else 0.)
    monkeypatch.setattr(module, 'encode_batch', encode)
    batches = module.benchmark_checked(None, None, {'G': [[1] * 10, [1] * 20], 'E': [None, None]}, tmp_path)
    assert batches['256'] != 4
    records = json.loads((tmp_path / 'batch_benchmark.json').read_text())
    assert records['tolerance'] == {'atol': 1e-4, 'rtol': 1e-4}
    assert next(r for r in records['measurements'] if r['batch'] == 4)['status'] == 'rejected_numerical_consistency'
