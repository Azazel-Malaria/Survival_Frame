from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import pandas as pd
import torch


SRC = Path(__file__).resolve().parents[2] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wsi_datasets.unified_survival import (
    UnifiedSurvivalDataset,
    _read_h5_dense_x,
    singleton_collate,
)


class StarpathJointDatasetTests(unittest.TestCase):
    def test_st_reader_rejects_same_width_with_swapped_gene_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'swapped_stpath_pred.h5ad'
            with h5py.File(path, 'w') as handle:
                handle.create_dataset(
                    'X', data=np.zeros((3, 2), dtype=np.float32)
                )
                var = handle.create_group('var')
                var.attrs['_index'] = '_index'
                var.create_dataset(
                    '_index', data=np.asarray([b'ST1', b'ST0'])
                )

            with self.assertRaisesRegex(ValueError, 'columns/order'):
                _read_h5_dense_x(str(path), ('ST0', 'ST1'))

    def test_st_reader_rejects_source_coordinates_that_are_out_of_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'coords_stpath_pred.h5ad'
            with h5py.File(path, 'w') as handle:
                handle.create_dataset(
                    'X', data=np.zeros((3, 2), dtype=np.float32)
                )
                var = handle.create_group('var')
                var.attrs['_index'] = '_index'
                var.create_dataset(
                    '_index', data=np.asarray([b'ST0', b'ST1'])
                )
                obsm = handle.create_group('obsm')
                obsm.create_dataset(
                    'source_coords',
                    data=np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.float64),
                )

            expected = np.asarray([[0, 0], [2, 2], [1, 1]], dtype=np.int64)
            with self.assertRaisesRegex(ValueError, 'out of order'):
                _read_h5_dense_x(
                    str(path), ('ST0', 'ST1'), expected_source_coords=expected
                )

    def test_constructed_dataset_indexes_st_and_exposes_joint_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wsi_dir = root / 'wsi'
            st_dir = root / 'st'
            wsi_dir.mkdir()
            st_dir.mkdir()
            slide_id = 'slide-a'

            with h5py.File(wsi_dir / f'{slide_id}.h5', 'w') as handle:
                handle.create_dataset(
                    'features', data=np.zeros((6, 768), dtype=np.float32)
                )
                coords = handle.create_dataset(
                    'coords', data=np.arange(12).reshape(6, 2)
                )
                coords.attrs['patch_size_level0'] = 256
            with h5py.File(
                st_dir / f'{slide_id}_stpath_pred.h5ad', 'w'
            ) as handle:
                handle.create_dataset('X', data=np.zeros((6, 2), dtype=np.float32))
                var = handle.create_group('var')
                var.attrs['_index'] = '_index'
                var.create_dataset('_index', data=np.asarray([b'ST0', b'ST1']))
                obsm = handle.create_group('obsm')
                obsm.create_dataset(
                    'source_coords', data=np.arange(12).reshape(6, 2)
                )

            signature_path = root / 'hallmarks.csv'
            pd.DataFrame({
                f'HALLMARK_{index:02d}': [f'G{index % 3}']
                for index in range(50)
            }).to_csv(signature_path, index=False)
            prototype_path = root / 'prototypes.pkl'
            with open(prototype_path, 'wb') as handle:
                pickle.dump(
                    {'prototypes': np.zeros((16, 768), dtype=np.float32)},
                    handle,
                )

            dataset = UnifiedSurvivalDataset(
                df_histo=pd.DataFrame({
                    'case_id': ['case-a'],
                    'slide_id': [slide_id],
                    'time': [10.0],
                    'censorship': [0],
                }),
                df_gene=pd.DataFrame({
                    'case_id': ['case-a'], 'G0': [0.0], 'G1': [1.0], 'G2': [2.0]
                }),
                data_source=[str(wsi_dir)],
                model_name='starpath',
                survival_time_col='time',
                censorship_col='censorship',
                n_label_bins=0,
                omics_layout='pathway',
                rna_normalization='starpath_dual',
                rna_set='surv_set',
                signature_path=str(signature_path),
                morphology_prototype_path=str(prototype_path),
                st_dir=str(st_dir),
                starpath_bag_size=4,
            )
            dataset.apply_scaler(dataset.get_scaler())
            item = dataset[0]

        self.assertEqual(dataset.st_gene_seq, ('ST0', 'ST1'))
        self.assertEqual(dataset.rna_gene_seq, ('G0', 'G1', 'G2'))
        self.assertEqual(len(dataset.pathway_names), 50)
        self.assertEqual(dataset.omic_sizes, [1] * 50)
        self.assertEqual(len(item['img']['slides'][0]), 4)
        self.assertEqual(len(item['img']['st'][0]), 4)
        self.assertEqual(tuple(item['img']['route_atlas'][0].shape), (16, 2))
        self.assertEqual(
            item['img']['route_atlas_counts'][0].tolist(), [6] + [0] * 15
        )
        self.assertEqual(
            item['img']['route_atlas_valid'][0].tolist(), [True] + [False] * 15
        )
        torch.testing.assert_close(
            item['img']['route_atlas'][0][1:], torch.zeros(15, 2)
        )
        self.assertEqual(len(item['omics']['coarse_pathways']), 50)
        self.assertEqual(tuple(item['omics']['fine_rna'].shape), (3,))

    def test_full_slide_atlas_is_built_before_sampling(self):
        size = 18
        patch_ids = torch.arange(size)
        anchor_ids = torch.div(patch_ids, 6, rounding_mode='floor')
        centroids = torch.zeros(16, 768)
        centroids[:, 0] = 100.0 * torch.arange(16)
        features = centroids.index_select(0, anchor_ids).clone()
        features[:, 1] = patch_ids.float()
        coords = torch.stack([patch_ids, patch_ids + 100], dim=1)
        st = np.stack([np.arange(size), np.arange(size) + 200], axis=1).astype(
            np.float32
        )
        rows = pd.DataFrame({'slide_id': ['slide-a']})

        def make_dataset():
            dataset = UnifiedSurvivalDataset.__new__(UnifiedSurvivalDataset)
            dataset.feature_paths = {'slide-a': 'unused-wsi'}
            dataset.st_paths = {'slide-a': 'unused-st'}
            dataset.st_gene_seq = ('ST0', 'ST1')
            dataset._validated_st_paths = set()
            dataset.morphology_centroids = centroids
            dataset._morphology_summary_cache = {}
            dataset.starpath_bag_size = 5
            dataset.is_training = True
            dataset.sampling_seed = 13
            dataset.split_name = 'fold=0:train'
            dataset._load_wsi = lambda path: (
                features.clone(), coords.clone(), 256
            )
            return dataset

        expected_atlas = torch.tensor([
            [2.5, 202.5],
            [8.5, 208.5],
            [14.5, 214.5],
        ])
        with mock.patch(
            'wsi_datasets.unified_survival._read_h5_dense_x',
            return_value=st,
        ):
            np.random.seed(3)
            first = make_dataset()._starpath_payload('case-a', rows)
            np.random.seed(99)
            second = make_dataset()._starpath_payload('case-a', rows)

        self.assertFalse(torch.equal(first['slides'][0], second['slides'][0]))
        for key in (
            'route_atlas', 'route_atlas_counts', 'route_atlas_valid'
        ):
            torch.testing.assert_close(first[key][0], second[key][0])
        torch.testing.assert_close(
            first['route_atlas'][0][:3], expected_atlas,
            rtol=0.0, atol=0.0,
        )
        torch.testing.assert_close(
            first['route_atlas'][0][3:], torch.zeros(13, 2),
            rtol=0.0, atol=0.0,
        )
        self.assertEqual(
            first['route_atlas_counts'][0].tolist(), [6, 6, 6] + [0] * 13
        )
        self.assertEqual(
            first['route_atlas_valid'][0].tolist(),
            [True, True, True] + [False] * 13,
        )
        self.assertEqual(len(first['slides'][0]), 5)
        self.assertEqual(len(first['st'][0]), 5)

    def test_dual_rna_views_share_one_raw_train_fold_table(self):
        dataset = UnifiedSurvivalDataset.__new__(UnifiedSurvivalDataset)
        raw = pd.DataFrame(
            [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]],
            index=['case-a', 'case-b'],
            columns=['G0', 'G1', 'G2'],
            dtype=np.float32,
        )
        dataset.requires_omics = True
        dataset.model_name = 'starpath'
        dataset.rna_normalization = 'standard'
        dataset.rna_set = 'surv_set'
        dataset.rna_gene_seq = tuple(raw.columns)
        dataset._starpath_raw_omics_data = raw.copy(deep=True)
        dataset.omics_data = raw.copy(deep=True)
        dataset.coarse_omics_data = raw.copy(deep=True)
        dataset.fine_omics_data = raw.copy(deep=True)
        dataset.omic_names = [['G0', 'G2'], ['G1']]

        scaler = dataset.get_scaler()
        dataset.apply_scaler(scaler)
        omics = dataset._omics_for_case('case-a')

        self.assertEqual(scaler['kind'], 'starpath_dual')
        self.assertEqual(set(omics), {'coarse_pathways', 'fine_rna'})
        self.assertEqual(len(omics['coarse_pathways']), 2)
        torch.testing.assert_close(
            omics['coarse_pathways'][0], torch.tensor([-1.0, -1.0])
        )
        # Fine uses one global range, while exact raw zeros retain the missing
        # value convention rather than becoming -1.
        torch.testing.assert_close(
            omics['fine_rna'], torch.tensor([0.0, -0.6, -0.2])
        )
        pd.testing.assert_frame_equal(dataset._starpath_raw_omics_data, raw)

        collated = singleton_collate([{
            'img': {'slides': [torch.zeros(2, 3)]},
            'omics': omics,
            'survival_time': torch.tensor([1.0]),
            'censorship': torch.tensor([0.0]),
            'label': torch.tensor([1]),
        }])
        self.assertEqual(tuple(collated['omics']['fine_rna'].shape), (1, 3))
        self.assertEqual(
            tuple(collated['omics']['coarse_pathways'][0].shape), (1, 2)
        )

    def test_mmp_rna_uses_one_per_gene_standardized_view_for_both_stages(self):
        dataset = UnifiedSurvivalDataset.__new__(UnifiedSurvivalDataset)
        raw = pd.DataFrame(
            [[1.0, 10.0, 100.0], [3.0, 30.0, 300.0]],
            index=['case-a', 'case-b'],
            columns=['G0', 'G1', 'G2'],
            dtype=np.float32,
        )
        dataset.requires_omics = True
        dataset.model_name = 'starpath'
        dataset.rna_normalization = 'standard'
        dataset.rna_set = 'mmp_set'
        dataset.rna_gene_seq = tuple(raw.columns)
        dataset._starpath_raw_omics_data = raw.copy(deep=True)
        dataset.omics_data = raw.copy(deep=True)
        dataset.coarse_omics_data = raw.copy(deep=True)
        dataset.fine_omics_data = raw.copy(deep=True)
        dataset.omic_names = [['G0', 'G2'], ['G1']]

        scaler = dataset.get_scaler()
        dataset.apply_scaler(scaler)
        omics = dataset._omics_for_case('case-a')

        self.assertEqual(scaler['kind'], 'starpath_mmp_standard')
        self.assertEqual(scaler['rna_set'], 'mmp_set')
        self.assertIs(dataset.coarse_omics_data, dataset.fine_omics_data)
        self.assertIs(dataset.coarse_omics_data, dataset.omics_data)
        torch.testing.assert_close(
            omics['coarse_pathways'][0], torch.tensor([-1.0, -1.0])
        )
        torch.testing.assert_close(
            omics['fine_rna'], torch.tensor([-1.0, -1.0, -1.0])
        )
        pd.testing.assert_frame_equal(dataset._starpath_raw_omics_data, raw)

    def test_starpath_scaler_rejects_reordered_gene_axis_across_splits(self):
        train = UnifiedSurvivalDataset.__new__(UnifiedSurvivalDataset)
        raw = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=['case-a', 'case-b'],
            columns=['G0', 'G1'],
            dtype=np.float32,
        )
        train.requires_omics = True
        train.model_name = 'starpath'
        train.rna_normalization = 'standard'
        train.rna_set = 'mmp_set'
        train.rna_gene_seq = tuple(raw.columns)
        train._starpath_raw_omics_data = raw
        scaler = train.get_scaler()

        validation = UnifiedSurvivalDataset.__new__(UnifiedSurvivalDataset)
        validation.requires_omics = True
        validation.model_name = 'starpath'
        validation.rna_normalization = 'standard'
        validation.rna_set = 'mmp_set'
        validation.rna_gene_seq = ('G1', 'G0')
        validation._starpath_raw_omics_data = raw[['G1', 'G0']]
        with self.assertRaisesRegex(ValueError, 'columns/order'):
            validation.apply_scaler(scaler)

    def test_full_slide_summary_then_joint_patch_sampling(self):
        size = 18
        patch_ids = torch.arange(size)
        features = torch.zeros(size, 768)
        features[:, 1] = patch_ids.float()
        coords = torch.stack([patch_ids, patch_ids + 100], dim=1)
        st = np.stack([np.arange(size), np.arange(size) + 200], axis=1).astype(
            np.float32
        )

        dataset = UnifiedSurvivalDataset.__new__(UnifiedSurvivalDataset)
        dataset.feature_paths = {'slide-a': 'unused-wsi'}
        dataset.st_paths = {'slide-a': 'unused-st'}
        dataset.st_gene_seq = ('ST0', 'ST1')
        dataset._validated_st_paths = set()
        dataset.morphology_centroids = torch.zeros(16, 768)
        dataset._morphology_summary_cache = {}
        dataset.starpath_bag_size = 5
        dataset.is_training = False
        dataset.sampling_seed = 13
        dataset.split_name = 'fold=0:test'
        dataset._load_wsi = lambda path: (features.clone(), coords.clone(), 256)
        summary_input_sizes = []

        actual_summary = dataset._full_slide_morphology_route_summary

        def summarize(slide_id, full_features, full_st):
            summary_input_sizes.append((len(full_features), len(full_st)))
            return actual_summary(slide_id, full_features, full_st)

        dataset._full_slide_morphology_route_summary = summarize
        rows = pd.DataFrame({'slide_id': ['slide-a']})
        with mock.patch(
            'wsi_datasets.unified_survival._read_h5_dense_x', return_value=st
        ) as reader:
            payload = dataset._starpath_payload('case-a', rows)

        self.assertEqual(summary_input_sizes, [(size, size)])
        self.assertEqual(len(payload['slides'][0]), 5)
        feature_ids = payload['slides'][0][:, 1].long()
        torch.testing.assert_close(feature_ids, payload['coords'][0][:, 0].long())
        torch.testing.assert_close(feature_ids, payload['st'][0][:, 0].long())
        np.testing.assert_array_equal(
            reader.call_args.kwargs['expected_source_coords'], coords.numpy()
        )
        self.assertEqual(
            set(payload),
            {
                'slides', 'coords', 'st', 'morphology_tokens',
                'morphology_occupancy', 'morphology_valid', 'slide_ids',
                'patch_sizes', 'route_atlas', 'route_atlas_counts',
                'route_atlas_valid',
            },
        )
        for key in ('route_atlas', 'route_atlas_counts', 'route_atlas_valid'):
            self.assertIn(key, payload)


if __name__ == '__main__':
    unittest.main()
