"""Survival contracts and regressions for the migrated baseline models."""

import pickle
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mil_models.components import process_surv
from mil_models.model_dimaf import DIMAF, DistanceCorrelation
from mil_models.model_factory import (
    create_multimodal_survival_model, _embedding_cache_metadata,
    _embedding_cache_path, _valid_embedding_payload, prepare_emb,
)
from mil_models.model_slotspe import (
    IterativeCrossAttention, SlotSPE, build_pathway_gene_indices,
)
from mil_models.survival_adapter import UnifiedSurvivalAdapter
from utils.losses import CoxLoss, NLLSurvLoss


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.signature = self.root / "signatures.csv"
        self.signature.write_text("P1,P2\nG2,G3\nG1,G2\n")
        self.composition = self.root / "composition.csv"
        self.composition.write_text("gene,P1,P2\nG1,1,0\nG2,1,1\nG3,0,1\n")

    def args(self, name, loss):
        return SimpleNamespace(
            survival_model=name, loss_fn=loss, n_label_bins=4, aux_nll_bins=4,
            feat_dim=8, in_dim=9, n_proto=16, omic_dim=3,
            rna_gene_seq=["G1", "G2", "G3"], nll_alpha=0.5,
            signature_path=str(self.signature), composition_path=str(self.composition),
            model_mm_type={"mmp_ot": "coattn_mot", "survpath": "survpath"}.get(name, "coattn"),
            model_histo_type="PANTHER", histo_agg="mean", append_embed="random",
            num_coattn_layers=1, net_indiv=False,
        )

    def assert_finite_backward(self, model, value):
        self.assertTrue(bool(torch.isfinite(value)))
        value.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        self.assertTrue(any(bool(g.abs().sum() > 0) for g in gradients))

    def test_factory_backbones_support_nll_and_cox(self):
        names = ["abmil", "transmil", "mcat", "mlp", "snn", "s_mlp",
                 "titan", "dimaf", "mmp_trans", "mmp_ot", "survpath"]
        for name in names:
            for loss in ("nll", "cox"):
                with self.subTest(model=name, loss=loss):
                    sizes = [3] * (50 if name == "dimaf" else 6 if name == "mcat" else 3)
                    model = create_multimodal_survival_model(self.args(name, loss), sizes)
                    batch_size = 2 if model.supports_batched_patients else 1
                    data = torch.randn(batch_size, 9, 8)
                    omics = [torch.randn(batch_size, size) for size in sizes]
                    if name == "dimaf":
                        data = torch.randn(batch_size, 16, 9)
                    elif name in {"mmp_trans", "mmp_ot"}:
                        data = torch.randn(batch_size, 16, 17)
                    elif name == "titan":
                        data = torch.randn(batch_size, 2, 768)
                    elif name in {"mlp", "snn", "s_mlp"}:
                        omics = torch.randn(batch_size, 3)
                    deferred = loss == "cox" and batch_size == 1
                    result, _ = model(
                        data, omics, label=torch.arange(batch_size) % 4,
                        survival_time=torch.arange(batch_size).float() + 10,
                        censorship=torch.zeros(batch_size),
                        loss_fn=NLLSurvLoss(alpha=0.5) if loss == "nll" else CoxLoss(),
                        defer_survival_loss=deferred,
                    )
                    self.assertEqual(result["logits"].shape, (batch_size, 4 if loss == "nll" else 1))
                    if deferred:
                        # The outer trainer builds a joint risk set from patients.
                        logits = torch.cat([result["logits"], result["logits"] * -0.5])
                        objective = CoxLoss()(logits, torch.tensor([10., 20.]), torch.zeros(2))["loss"]
                    else:
                        objective = result["loss"]
                    self.assert_finite_backward(model, objective)

    def small_slotspe(self, loss):
        backbone = SlotSPE(
            input_dim=8, n_classes=1 if loss == "cox" else 4, aux_nll_bins=4,
            signature_path=str(self.signature), rna_gene_seq=["G1", "G2", "G3"],
            wsi_projection_dim=16, slot_iters=2, cross_iters=2,
        )
        return UnifiedSurvivalAdapter(backbone, "slotspe", aux_nll_alpha=0.5)

    def test_slotspe_cox_retains_discrete_auxiliary_heads(self):
        model = self.small_slotspe("cox")
        result, _ = model(
            torch.randn(1, 9, 8), torch.randn(1, 3),
            label=torch.tensor([300.]), discrete_label=torch.tensor([3]),
            survival_time=torch.tensor([300.]), censorship=torch.tensor([0.]),
            loss_fn=CoxLoss(), defer_survival_loss=True,
        )
        self.assertEqual(result["logits"].shape, (1, 1))
        self.assertEqual(result["logits_wsi"].shape, (1, 4))
        self.assertEqual(result["logits_omic"].shape, (1, 4))
        self.assert_finite_backward(model, result["auxiliary_loss"])
        self.assertGreater(float(model.backbone.slot_decoder_wsi.decoder.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.backbone.slot_decoder_omic.decoder.weight.grad.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "discrete_label"):
            model(torch.randn(1, 9, 8), torch.randn(1, 3),
                  label=torch.tensor([300.]), survival_time=torch.tensor([300.]),
                  censorship=torch.tensor([0.]), loss_fn=CoxLoss(), defer_survival_loss=True)

    def test_slotspe_nll_and_official_missing_rna_reconstruction(self):
        model = self.small_slotspe("nll")
        result, _ = model(torch.randn(1, 9, 8), torch.randn(1, 3),
                          label=torch.tensor([3]), censorship=torch.tensor([0.]),
                          loss_fn=NLLSurvLoss(alpha=0.5))
        self.assert_finite_backward(model, result["loss"])
        model.eval()
        reconstructed, _ = model(torch.randn(1, 9, 8), None, loss_fn=NLLSurvLoss())
        self.assertEqual(reconstructed["risk"].shape, (1, 1))
        self.assertTrue(bool(torch.isfinite(reconstructed["risk"]).all()))

    def test_padding_does_not_change_patient_prediction(self):
        model = self.small_slotspe("nll").eval()
        data, rna = torch.randn(1, 9, 8), torch.randn(1, 3)
        torch.manual_seed(41)
        reference, _ = model(data, rna, loss_fn=NLLSurvLoss())
        padded = torch.cat([data, torch.full((1, 3, 8), 1e6)], dim=1)
        torch.manual_seed(41)
        actual, _ = model(padded, rna, attn_mask=torch.tensor([[1] * 9 + [0] * 3]),
                          loss_fn=NLLSurvLoss())
        torch.testing.assert_close(reference["logits"], actual["logits"])

    def test_no_fabricated_pathways_or_silent_rna_axis_changes(self):
        names, indices = build_pathway_gene_indices(["G3", "G2", "G1"], str(self.signature))
        self.assertEqual(indices, [[2, 1], [1, 0]])
        with self.assertRaises(FileNotFoundError):
            build_pathway_gene_indices(["G1"], str(self.root / "absent.csv"))
        model = self.small_slotspe("nll")
        for values in (torch.zeros(1, 2), torch.tensor([[1., float("nan"), 3.]])):
            with self.assertRaises(ValueError):
                model(torch.randn(1, 9, 8), values)

    def test_static_kv_matches_official_release_golden_output(self):
        # Generated with official models/transformer.py at commit
        # 02051a2083add7b427727e0200e4263516903561. The previous adaptation
        # normalized static K/V twice; this fixed reference catches that drift.
        model = IterativeCrossAttention(dim=8, num_heads=2, iters=2).eval()
        generator = torch.Generator().manual_seed(39)
        model.load_state_dict({key: torch.randn(value.shape, generator=generator) * .15
                               for key, value in sorted(model.state_dict().items())})
        x = torch.linspace(-2, 3, 24).reshape(1, 3, 8)
        y = torch.linspace(3, -1, 16).reshape(1, 2, 8)
        expected = torch.tensor([
            -1.134090424, -.405260235, -.791575611, .032621987, -.240003645,
            -.414242417, -.398206323, -.568759382, -.096114963, .000416063,
            -.277908951, .179844961, .028797790, -.055102021, .237978518,
            .124819927, .469228953, .597595811, 1.023911953, .235075787,
            1.170660853, .980746269, 1.121378541, .512199640, 1.217407942,
            1.357982635, .827357590, .535932183, -.255511820, .205372676,
            .363781244, .501878321, .295874923, .395990193, .080324218,
            -.069197387, -.607150912, -.077428490, -.024119170, -.609637916,
        ]).reshape(1, 5, 8)
        actual = torch.cat(model(x, y), dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_dimaf_singleton_distance_correlation_is_explicit_zero(self):
        x, y = torch.randn(1, 8, requires_grad=True), torch.randn(1, 8, requires_grad=True)
        loss = DistanceCorrelation()(x, y)
        loss.backward()
        self.assertEqual(float(loss), 0)
        torch.testing.assert_close(x.grad, torch.zeros_like(x))
        torch.testing.assert_close(y.grad, torch.zeros_like(y))

    def test_cox_rejects_discrete_label_as_continuous_time(self):
        with self.assertRaisesRegex(ValueError, "continuous survival time"):
            process_surv(torch.zeros(2, 1), torch.tensor([0, 1]), torch.zeros(2), CoxLoss())

    def test_cache_identity_rejects_different_prototype_order_or_source(self):
        prototype = self.root / "prototype.pkl"
        prototype.write_bytes(b"first prototype")
        feature = self.root / "slide.pt"
        feature.write_bytes(b"feature data")
        dataset = SimpleNamespace(
            idx2sample_df=pd.DataFrame({"sample_id": ["A", "B"]}),
            data_df=pd.DataFrame({"case_id": ["A", "B"], "slide_id": ["S1", "S2"]}),
            feature_paths={"S1": str(feature)}, bag_size=-1, sampling_seed=1,
            survival_time_labels=torch.tensor([10., 20.]), censorship_labels=torch.tensor([0., 1.]),
            disc_labels=torch.tensor([0, 3]),
        )
        loaders = {"train": SimpleNamespace(dataset=dataset)}
        args = self.args("dimaf", "nll")
        args.model_histo_config = "PANTHER_default"
        args.load_proto, args.fix_proto = True, True
        args.proto_path, args.embedding_cache_dir = str(prototype), str(self.root / "cache")
        args.data_source = [str(self.root)]
        args.seed, args.out_type, args.tau, args.ot_eps, args.em_iter = 1, "allcat", .001, .1, 1
        metadata = _embedding_cache_metadata(loaders, args, "survival")
        original_path = _embedding_cache_path(args, metadata)
        prototype.write_bytes(b"other prototype")
        self.assertNotEqual(original_path, _embedding_cache_path(args, _embedding_cache_metadata(loaders, args, "survival")))
        prototype.write_bytes(b"first prototype")
        dataset.idx2sample_df = dataset.idx2sample_df.iloc[::-1]
        self.assertNotEqual(original_path, _embedding_cache_path(args, _embedding_cache_metadata(loaders, args, "survival")))
        dataset.idx2sample_df = dataset.idx2sample_df.iloc[::-1]
        args.data_source = [str(self.root / "other")]
        self.assertNotEqual(original_path, _embedding_cache_path(args, _embedding_cache_metadata(loaders, args, "survival")))
        payload = {"metadata": metadata, "embeddings": {"train": {
            "sample_ids": ["A", "B"], "X": torch.zeros(2, 16, 9), "y": np.zeros(2),
        }}}
        self.assertTrue(_valid_embedding_payload(payload, metadata))
        payload["embeddings"]["train"]["sample_ids"] = ["B", "A"]
        self.assertFalse(_valid_embedding_payload(payload, metadata))

    def test_panther_cache_round_trip_preserves_rows(self):
        class Dataset:
            def __init__(self):
                self.idx2sample_df = pd.DataFrame({"sample_id": ["A", "B"]})
                self.features = torch.randn(2, 20, 8)
                self.calls = 0
            def __len__(self):
                return 2
            def __getitem__(self, index):
                self.calls += 1
                return {"img": self.features[index], "label": torch.tensor(index),
                        "censorship": torch.tensor(float(index)),
                        "survival_time": torch.tensor(float(index + 10))}

        prototype = self.root / "prototype.pkl"
        with prototype.open("wb") as handle:
            pickle.dump({"prototypes": np.random.default_rng(3).normal(size=(1, 16, 8)).astype("float32")}, handle)
        args = self.args("dimaf", "nll")
        args.in_dim = 8
        args.model_histo_config = "PANTHER_default"
        args.load_proto, args.fix_proto = True, True
        args.proto_path, args.embedding_cache_dir = str(prototype), str(self.root / "cache")
        args.data_source = [str(self.root)]
        args.seed, args.out_type, args.tau, args.ot_eps, args.em_iter = 1, "allcat", .001, .1, 1
        dataset = Dataset()
        loaders = {"train": SimpleNamespace(dataset=dataset)}
        _, path = prepare_emb(loaders, args, mode="survival")
        self.assertEqual(dataset.calls, 2)
        self.assertEqual(dataset.X.shape, (2, 16 * 17))
        first = dataset.X.clone()
        dataset.X = None
        _, cached_path = prepare_emb(loaders, args, mode="survival")
        self.assertEqual(path, cached_path)
        self.assertEqual(dataset.calls, 2)
        torch.testing.assert_close(first, dataset.X)


if __name__ == "__main__":
    unittest.main()
