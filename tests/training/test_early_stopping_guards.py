import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from utils.checkpoint import CheckpointManager
from training.trainer import train
from types import SimpleNamespace


class CheckpointTests(unittest.TestCase):
    def test_last_and_best_are_independent_of_stopping(self):
        for selection in ('last', 'best'):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as tmp:
                model = nn.Linear(1, 1, bias=False)
                manager = CheckpointManager(tmp, selection=selection, early_stopping=True,
                                            patience=1, min_epochs=2)
                with torch.no_grad():
                    model.weight.fill_(1)
                self.assertFalse(manager.step(0, model, {'c_index': .8, 'loss': .2}))
                with torch.no_grad():
                    model.weight.fill_(2)
                self.assertTrue(manager.step(1, model, {'c_index': .6, 'loss': .4}))
                record = manager.finish(model, 1)
                self.assertEqual(model.weight.item(), 1 if selection == 'best' else 2)
                self.assertEqual(record['best_epoch'], 0)
                self.assertEqual(record['last_epoch'], 1)
                self.assertEqual(record['stop_reason'], 'patience')
                self.assertTrue((Path(tmp) / 'last_checkpoint.pth').is_file())
                self.assertTrue((Path(tmp) / 'best_checkpoint.pth').is_file())

    def test_best_runs_without_early_stopping_and_cindex_tie_selects_later(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = nn.Linear(1, 1)
            manager = CheckpointManager(tmp, selection='best', early_stopping=False)
            for epoch in range(3):
                self.assertFalse(manager.step(epoch, model, {'c_index': .7, 'loss': 1.0}))
            self.assertEqual(manager.finish(model, 2)['selected_epoch'], 2)

    def test_missing_or_foreign_best_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = nn.Linear(1, 1)
            manager = CheckpointManager(tmp, selection='best')
            with self.assertRaisesRegex(RuntimeError, 'requires finite validation'):
                manager.finish(model, 0)
            manager.step(0, model, {'c_index': .7, 'loss': .2})
            file = Path(tmp) / 'best_checkpoint.pth'
            state = torch.load(file, weights_only=False)
            state['run_id'] = 'foreign-run'
            torch.save(state, file)
            with self.assertRaisesRegex(RuntimeError, 'provenance'):
                manager.finish(model, 1)

    def test_invalid_validation_metric_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp)
            with self.assertRaisesRegex(ValueError, 'finite'):
                manager.step(0, nn.Linear(1, 1), {'c_index': float('nan')})

    def test_selection_or_stopping_requires_validation(self):
        for selection, early_stop in [('best', 0), ('last', 1)]:
            args = SimpleNamespace(checkpoint_selection=selection, early_stopping=early_stop)
            with self.assertRaisesRegex(ValueError, 'validation split'):
                train({'train': object(), 'test': object()}, args)


if __name__ == '__main__':
    unittest.main()
