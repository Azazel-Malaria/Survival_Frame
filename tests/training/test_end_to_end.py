"""Small real training runs cover labels, risk sets, prototypes and checkpoints."""
import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
import torch

from training.main_survival import build_parser, main
from utils.experiment_config import (load_data_paths, resolve_protocol, prototype_spec,
                                     file_sha256)


class TrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root):
        rng = np.random.default_rng(42)
        feature = root / 'features/BRCA'
        feature.mkdir(parents=True)
        rows, genes = [], []
        for i in range(12):
            group = 'AA' if i < 8 else 'BB' if i < 10 else 'CC'
            case = f'TCGA-{group}-{i:04d}'
            slide = case + '-01Z-00-DX1'
            rows.append({'case_id': case, 'slide_id': slide, 'tissue_source_site': group,
                         'dss_survival_days': float(i + 1), 'dss_censorship': 0,
                         'os_survival_days': float(2 * i + 1), 'os_censorship': 0})
            genes.append({'case_id': case, 'G1': float(i + 1), 'G2': float((i % 3) + 1)})
            with h5py.File(feature / (slide + '.h5'), 'w') as f:
                f['features'] = rng.normal(size=(20, 768)).astype('float32')
                coords = f.create_dataset('coords', data=np.arange(40).reshape(20, 2))
                coords.attrs['patch_size_level0'] = 512
        data = pd.DataFrame(rows)
        paths = {
            'rna_root': str(root / 'rna'), 'feature_root': str(root / 'features'),
            'st_root': str(root / 'st'), 'prototype_root': str(root / 'prototypes'),
            'results_root': str(root / 'results'), 'titan_root': str(root / 'titan'),
            'starpath_titan_root': str(root / 'private_titan'), 'split_roots': {},
        }
        for endpoint in ('dss', 'os'):
            for mode in ('train_test', 'train_val_test'):
                folder = root / endpoint / mode
                paths['split_roots'][endpoint + '/' + mode] = str(folder)
                split = folder / 'TCGA_BRCA_overall_survival_k=0'
                split.mkdir(parents=True)
                data.iloc[:8 if mode == 'train_val_test' else 10].to_csv(split / 'train.csv', index=False)
                data.iloc[10:].to_csv(split / 'test.csv', index=False)
                if mode == 'train_val_test':
                    data.iloc[8:10].to_csv(split / 'val.csv', index=False)
        for rel in ('surv_set/raw_rna_data/combine/brca', 'mmp_set/hallmarks/BRCA'):
            folder = root / 'rna' / rel
            folder.mkdir(parents=True)
            pd.DataFrame(genes).to_csv(folder / 'rna_clean.csv', index=False)
        metadata = root / 'rna/mmp_set/metadata'
        metadata.mkdir(parents=True)
        pd.DataFrame({f'P{i}': ['G1' if i % 2 else 'G2'] for i in range(50)}).to_csv(
            metadata / 'hallmarks_signatures.csv', index=False)
        config = root / 'paths.json'
        config.write_text(json.dumps(paths))
        return config

    def run_model(self, root, config, model, loss, *, selection='last', early=False):
        output = root / f'output_{model}_{loss}_{selection}_{early}'
        values = ['--data_paths', str(config), '--survival_model', model, '--cancer_type', 'BRCA',
                  '--fold', '0', '--loss_fn', loss, '--results_dir', str(output),
                  '--num_workers', '0', '--max_epochs', '2', '--checkpoint_selection', selection,
                  '--n_label_bins', '2', '--print_every', '100', '--early_stopping', str(int(early)),
                  '--es_min_epochs', '1', '--es_patience', '1', '--warmup_epochs', '0']
        if model == 'dimaf':
            paths = load_data_paths(config)
            protocol = resolve_protocol(paths=paths)
            spec = prototype_spec(protocol, 'BRCA', 0, paths)
            proto = Path(spec['path'])
            proto.parent.mkdir(parents=True)
            with proto.open('wb') as f:
                pickle.dump({'prototypes': np.random.default_rng(1).normal(size=(1, 16, 768)).astype('float32')}, f)
            meta = dict(spec['metadata'], prototype_sha256=file_sha256(proto))
            Path(spec['metadata_path']).write_text(json.dumps(meta))
        args = build_parser().parse_args(values)
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(args)
        self.assertTrue(np.isfinite(result['c_index_test']))
        record = json.loads((output / 'checkpoint_selection.json').read_text())
        self.assertEqual(record['checkpoint_selection'], selection)
        self.assertTrue((output / 'last_checkpoint.pth').is_file())
        with open(output / 'data_transforms.pkl', 'rb') as f:
            transform = pickle.load(f)
        if loss == 'nll':
            self.assertEqual(len(transform['label_bins']), 3)
        self.assertEqual(args.batch_size, 64 if loss == 'cox' or model == 'dimaf' else 1)
        if early or selection == 'best':
            self.assertEqual(args.split_mode, 'train_val_test')
            self.assertTrue((output / 'best_checkpoint.pth').is_file())
        return record

    def test_training_default_last_best_and_early_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            self.run_model(root, config, 'mlp', 'nll')
            self.run_model(root, config, 'mlp', 'nll', selection='best')
            self.run_model(root, config, 'mlp', 'cox', early=True)

    def test_single_patient_cox_and_dimaf_prototype_nll_batch64(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            self.run_model(root, config, 'abmil', 'cox')
            self.run_model(root, config, 'dimaf', 'nll')


if __name__ == '__main__':
    unittest.main()
