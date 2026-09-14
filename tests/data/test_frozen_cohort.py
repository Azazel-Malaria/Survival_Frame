import unittest
import numpy as np
import pandas as pd
import torch
from wsi_datasets.unified_survival import UnifiedSurvivalDataset
from wsi_datasets.survival_sampling import EventAwareRiskSetBatchSampler


class FrozenCohortTests(unittest.TestCase):
    def dataset(self, times, genes=None, *, training=True, bins=None):
        n = len(times)
        histo = pd.DataFrame({'case_id': [f'case-{i}' for i in range(n)],
                              'slide_id': [f'slide-{i}' for i in range(n)],
                              'dss_survival_days': times, 'dss_censorship': [0] * n})
        if genes is None:
            genes = pd.DataFrame({'case_id': histo.case_id, 'G1': np.arange(n), 'G2': np.arange(n) * 2})
        return UnifiedSurvivalDataset(df_histo=histo, df_gene=genes, data_source=[],
                                      model_name='mlp', survival_time_col='dss_survival_days',
                                      censorship_col='dss_censorship', n_label_bins=2,
                                      label_bins=bins, is_training=training,
                                      omics_layout='flat', rna_normalization='standard')

    def test_missing_rna_cannot_silently_reduce_a_frozen_cohort(self):
        genes = pd.DataFrame({'case_id': ['case-0', 'case-2'], 'G1': [1., 2.]})
        with self.assertRaisesRegex(ValueError, 'frozen split.*without configured RNA'):
            self.dataset([1., 2., 3.], genes)

    def test_validation_reuses_train_bins_and_scaler(self):
        train = self.dataset([1., 2., 3., 4.])
        scaler = train.get_scaler()
        bins = train.get_label_bins().copy()
        genes = pd.DataFrame({'case_id': ['case-0', 'case-1'], 'G1': [100., 200.], 'G2': [200., 400.]})
        val = self.dataset([2., 1000.], genes, training=False, bins=bins)
        val.apply_scaler(scaler)
        np.testing.assert_array_equal(val.get_label_bins(), bins)
        torch.testing.assert_close(val.disc_labels, torch.tensor([0, 1]))
        np.testing.assert_allclose(val.omics_data.to_numpy(), scaler['scaler'].transform(genes[['G1', 'G2']].to_numpy()))
        np.testing.assert_array_equal(scaler['scaler'].mean_, [1.5, 3.])

    def test_validation_cannot_fit_its_own_bins(self):
        with self.assertRaisesRegex(ValueError, 'fitted on training patients'):
            self.dataset([10., 20., 30.], training=False)

    def test_missing_times_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-finite survival labels'):
            self.dataset([1., float('nan'), 3.])

    def test_cox_tail_is_merged_without_dropping_or_repeating_patients(self):
        dataset = self.dataset(list(range(1, 10)))
        sampler = EventAwareRiskSetBatchSampler(dataset, riskset_size=4, seed=7)
        batches = list(sampler)
        self.assertEqual(sorted(len(batch) for batch in batches), [2, 3, 4])
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(9)))


if __name__ == '__main__':
    unittest.main()
