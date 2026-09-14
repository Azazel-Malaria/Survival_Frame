from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wsi_datasets.dataset_utils import apply_aligned_sampling
from wsi_datasets.unified_survival import UnifiedSurvivalDataset
from mil_models.modal_starpath import (
    summarize_full_slide_morphology_route_atlas,
)


def _morphology_centroids():
    centroids = torch.zeros(16, 768, dtype=torch.float32)
    centroids[:, 0] = torch.arange(16, dtype=torch.float32) * 100.0
    return centroids


def _patch_arrays(size=24):
    patch_ids = torch.arange(size)
    prototype_ids = torch.div(patch_ids, 2, rounding_mode="floor")
    features = torch.zeros(size, 768, dtype=torch.float32)
    features[:, 0] = prototype_ids.float() * 100.0
    features[:, 1] = patch_ids.float()
    coords = torch.stack([patch_ids, patch_ids + 1000], dim=1)
    metadata = np.stack(
        [np.arange(size), np.arange(size) + 2000], axis=1
    ).astype(np.float32)
    return features, coords, metadata


def _dataset_harness(is_training, cap):
    dataset = UnifiedSurvivalDataset.__new__(UnifiedSurvivalDataset)
    dataset.feature_paths = {"slide-a": "unused-feature-path"}
    dataset.morphology_centroids = _morphology_centroids()
    dataset._morphology_summary_cache = {}
    dataset.st_paths = {"slide-a": "unused-st-path"}
    dataset.st_gene_seq = ("ST0", "ST1")
    dataset._validated_st_paths = set()
    dataset.starpath_bag_size = cap
    dataset.is_training = is_training
    dataset.sampling_seed = 71
    dataset.split_name = "train" if is_training else "test"
    features, coords, st = _patch_arrays()
    dataset._load_wsi = lambda path: (features.clone(), coords.clone(), 256)
    build_payload = dataset._starpath_payload

    def load_payload(case_id, rows):
        with mock.patch("wsi_datasets.unified_survival._read_h5_dense_x", return_value=st):
            return build_payload(case_id, rows)

    dataset._starpath_payload = load_payload
    return dataset


def _assert_payload_is_aligned(test_case, payload):
    feature_ids = payload["slides"][0][:, 1].to(torch.int64)
    coord_ids = payload["coords"][0][:, 0].to(torch.int64)
    torch.testing.assert_close(feature_ids, coord_ids)
    torch.testing.assert_close(feature_ids, payload["st"][0][:, 0].to(torch.int64))
    for key in ("route_atlas", "route_atlas_counts", "route_atlas_valid"):
        test_case.assertIn(key, payload)
    test_case.assertEqual(len(payload["slide_ids"]), 1)
    for key in (
        "morphology_tokens",
        "morphology_occupancy",
        "morphology_valid",
    ):
        test_case.assertEqual(len(payload[key]), 1)


class StarpathPatchProtocolTests(unittest.TestCase):
    def test_single_occupied_anchor_has_exact_atlas_mean_and_empty_slots(self):
        centroids = _morphology_centroids()
        anchor_id = 7
        features = centroids[anchor_id].repeat(4, 1)
        st = torch.tensor([
            [1.0, 10.0, -2.0],
            [3.0, 20.0, 0.0],
            [5.0, 30.0, 2.0],
            [7.0, 40.0, 4.0],
        ])

        (
            morphology_tokens,
            morphology_occupancy,
            morphology_valid,
            route_atlas,
            route_counts,
            route_valid,
        ) = summarize_full_slide_morphology_route_atlas(
            features, st, centroids
        )

        self.assertEqual(tuple(route_atlas.shape), (16, 3))
        self.assertEqual(route_counts.dtype, torch.long)
        self.assertEqual(route_valid.dtype, torch.bool)
        self.assertEqual(route_counts.tolist(), [0] * 7 + [4] + [0] * 8)
        torch.testing.assert_close(
            route_atlas[anchor_id], st.mean(dim=0), rtol=0.0, atol=0.0
        )
        inactive = ~route_valid
        torch.testing.assert_close(
            route_atlas[inactive],
            torch.zeros_like(route_atlas[inactive]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            morphology_tokens[inactive],
            torch.zeros_like(morphology_tokens[inactive]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            morphology_occupancy,
            route_counts.to(torch.float32) / 4.0,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(morphology_valid, route_valid)

    def test_train_and_test_apply_the_same_per_slide_cap(self):
        rows = pd.DataFrame({"slide_id": ["slide-a"]})
        np.random.seed(3)
        train_payload = _dataset_harness(True, 5)._starpath_payload(
            "case-a", rows
        )
        test_payload = _dataset_harness(False, 5)._starpath_payload(
            "case-a", rows
        )

        self.assertEqual(len(train_payload["slides"][0]), 5)
        self.assertEqual(len(test_payload["slides"][0]), 5)
        _assert_payload_is_aligned(self, train_payload)
        _assert_payload_is_aligned(self, test_payload)

    def test_evaluation_sampling_is_stable_and_ignores_global_rng(self):
        rows = pd.DataFrame({"slide_id": ["slide-a"]})
        dataset = _dataset_harness(False, 5)
        first = dataset._starpath_payload("case-a", rows)
        np.random.seed(999)
        np.random.random(1000)
        second = dataset._starpath_payload("case-a", rows)

        for key in (
            "slides",
            "coords",
            "morphology_tokens",
            "morphology_occupancy",
            "morphology_valid",
            "route_atlas", "route_atlas_counts", "route_atlas_valid", "st",
        ):
            torch.testing.assert_close(first[key][0], second[key][0])
        _assert_payload_is_aligned(self, first)

    def test_full_slide_summary_is_independent_of_sampling(self):
        rows = pd.DataFrame({"slide_id": ["slide-a"]})
        np.random.seed(3)
        first = _dataset_harness(True, 5)._starpath_payload("case-a", rows)
        np.random.seed(99)
        second = _dataset_harness(True, 5)._starpath_payload("case-a", rows)
        full = _dataset_harness(True, -1)._starpath_payload("case-a", rows)

        self.assertFalse(torch.equal(first["slides"][0], second["slides"][0]))
        self.assertEqual(len(first["slides"][0]), 5)
        self.assertEqual(len(full["slides"][0]), 24)
        for key in (
            "morphology_tokens",
            "morphology_occupancy",
            "morphology_valid",
            "route_atlas", "route_atlas_counts", "route_atlas_valid",
        ):
            torch.testing.assert_close(first[key][0], second[key][0])
            torch.testing.assert_close(first[key][0], full[key][0])

        tokens = first["morphology_tokens"][0]
        occupancy = first["morphology_occupancy"][0]
        valid = first["morphology_valid"][0]
        self.assertEqual(tuple(tokens.shape), (16, 768))
        self.assertEqual(tuple(occupancy.shape), (16,))
        self.assertEqual(tuple(valid.shape), (16,))
        self.assertEqual(valid.dtype, torch.bool)
        torch.testing.assert_close(occupancy.sum(), torch.tensor(1.0))
        torch.testing.assert_close(
            occupancy[:12], torch.full((12,), 2.0 / 24.0)
        )
        torch.testing.assert_close(occupancy[12:], torch.zeros(4))
        self.assertTrue(bool(valid[:12].all()))
        self.assertFalse(bool(valid[12:].any()))
        torch.testing.assert_close(tokens[12:], torch.zeros(4, 768))

    def test_full_bag_option_and_explicit_indices_preserve_alignment(self):
        features, coords, metadata = _patch_arrays()
        full = apply_aligned_sampling(-1, features, coords, metadata)
        self.assertEqual(len(full[0]), len(features))

        sampled = apply_aligned_sampling(
            5,
            features,
            coords,
            metadata,
            indices=np.array([0, 2, 4, 6, 8]),
        )
        feature_ids = sampled[0][:, 1].to(torch.int64)
        torch.testing.assert_close(feature_ids, sampled[1][:, 0])
        torch.testing.assert_close(
            feature_ids, torch.from_numpy(sampled[2][:, 0]).to(torch.int64)
        )


class StarpathHallmarkGroupingTests(unittest.TestCase):
    @staticmethod
    def _grouping_harness(signature_path, available_genes):
        dataset = UnifiedSurvivalDataset.__new__(UnifiedSurvivalDataset)
        dataset.signature_path = str(signature_path)
        dataset.model_name = "starpath"
        dataset.omics_layout = "pathway"
        dataset.rna_gene_seq = tuple(available_genes)
        dataset.omic_names = []
        dataset.omic_sizes = []
        dataset.pathway_names = tuple()
        return dataset

    def test_preserves_fifty_signature_columns_as_variable_gene_vectors(self):
        columns = [f"HALLMARK_{index:02d}" for index in range(50)]
        signatures = pd.DataFrame({
            name: pd.Series([f"G{index}_A", f"G{index}_B"][: 1 + index % 2])
            for index, name in enumerate(columns)
        })
        available = {
            str(gene)
            for column in signatures.columns
            for gene in signatures[column].dropna()
        }
        with tempfile.TemporaryDirectory() as directory:
            signature_path = Path(directory) / "hallmarks.csv"
            signatures.to_csv(signature_path, index=False)
            dataset = self._grouping_harness(signature_path, available)
            dataset._build_gene_groups()

        self.assertEqual(dataset.pathway_names, tuple(columns))
        self.assertEqual(len(dataset.omic_names), 50)
        self.assertEqual(dataset.omic_sizes, [1 + index % 2 for index in range(50)])

    def test_rejects_non_fifty_or_empty_hallmark_group(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            short_path = directory / "short.csv"
            pd.DataFrame({f"P{index}": [f"G{index}"] for index in range(49)}).to_csv(
                short_path, index=False
            )
            short = self._grouping_harness(
                short_path, [f"G{index}" for index in range(49)]
            )
            with self.assertRaisesRegex(ValueError, "exactly 50"):
                short._build_gene_groups()

            empty_path = directory / "empty.csv"
            pd.DataFrame({
                f"P{index}": [f"G{index}"] for index in range(50)
            }).to_csv(empty_path, index=False)
            empty = self._grouping_harness(
                empty_path, [f"G{index}" for index in range(49)]
            )
            with self.assertRaisesRegex(ValueError, "has no genes"):
                empty._build_gene_groups()


if __name__ == "__main__":
    unittest.main()
