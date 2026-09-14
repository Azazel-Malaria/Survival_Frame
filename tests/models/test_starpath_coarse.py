import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mil_models.modal_starpath import (
    CoarseConditioner,
    HallmarkPathwayEncoder,
    plain_entropy_kl_uot,
    transported_molecular_context,
)
from mil_models.modal_starpath import MeanSlideAggregator
from mil_models.modal_starpath import summarize_full_slide_morphology
from mil_models.modal_starpath import STARPathPatientAdapter


def _assert_finite_nonzero(test_case, values, name):
    test_case.assertIsNotNone(values, f"{name} has no gradient")
    test_case.assertTrue(torch.isfinite(values).all(), f"{name} is not finite")
    test_case.assertGreater(values.abs().sum().item(), 0.0, f"{name} is zero")


def _omic_sizes():
    return [1 + index % 3 for index in range(50)]


def _centroids():
    centroids = torch.zeros(16, 768)
    centroids[:, 0] = torch.linspace(-3.0, 3.0, 16)
    centroids[:, 1] = torch.linspace(1.5, -1.5, 16)
    return centroids


def _sparse_slide(centroids, prototype_ids, *, requires_grad=False):
    ids = torch.tensor(prototype_ids, dtype=torch.long)
    features = centroids.index_select(0, ids).clone()
    features.requires_grad_(requires_grad)
    coords = torch.stack(
        [torch.arange(len(ids)), torch.arange(len(ids)) + 100], dim=1
    )
    tokens, occupancy, valid = summarize_full_slide_morphology(
        features.detach(), centroids
    )
    return features, coords, tokens, occupancy, valid


def _make_conditioner():
    return CoarseConditioner(
        omic_sizes=_omic_sizes(),
        pathway_names=[f"HALLMARK_{index:02d}" for index in range(50)],
        morphology_centroids=_centroids(),
        pathway_dim=8,
        compatibility_dim=7,
        context_dim=6,
        pathway_dropout=0.0,
        uot_iterations=20,
    )


class HallmarkPathwayEncoderTests(unittest.TestCase):
    def test_fifty_independent_two_block_variable_width_snns_receive_gradients(self):
        torch.manual_seed(3)
        sizes = [1 + index % 5 for index in range(50)]
        encoder = HallmarkPathwayEncoder(sizes, output_dim=7, dropout=0.0)

        self.assertEqual(len(encoder.networks), 50)
        self.assertEqual(len({id(network) for network in encoder.networks}), 50)
        weight_pointers = set()
        for index, (width, network) in enumerate(zip(sizes, encoder.networks)):
            self.assertEqual(len(network), 2)
            linear_layers = [
                module for module in network.modules() if isinstance(module, nn.Linear)
            ]
            self.assertEqual(len(linear_layers), 2)
            self.assertEqual(linear_layers[0].in_features, width)
            self.assertEqual(linear_layers[0].out_features, 7)
            self.assertEqual(linear_layers[1].in_features, 7)
            self.assertEqual(linear_layers[1].out_features, 7)
            for block in network:
                self.assertIsInstance(block[0], nn.Linear)
                self.assertIsInstance(block[1], nn.ELU)
                self.assertIsInstance(block[2], nn.AlphaDropout)
            pointer = linear_layers[0].weight.data_ptr()
            self.assertNotIn(pointer, weight_pointers, f"shared SNN weight at {index}")
            weight_pointers.add(pointer)

        pathways = [
            torch.randn(width, requires_grad=True) for width in sizes
        ]
        encoded = encoder(pathways)
        self.assertEqual(tuple(encoded.shape), (50, 7))
        encoded.square().sum().backward()

        for index, (values, network) in enumerate(zip(pathways, encoder.networks)):
            _assert_finite_nonzero(self, values.grad, f"pathway input {index}")
            network_gradients = [
                parameter.grad for parameter in network.parameters()
            ]
            self.assertTrue(all(gradient is not None for gradient in network_gradients))
            self.assertGreater(
                sum(gradient.abs().sum().item() for gradient in network_gradients),
                0.0,
                f"pathway SNN {index} received no gradient",
            )

    def test_rejects_49_pathways_and_incorrect_gene_vector_width(self):
        with self.assertRaisesRegex(ValueError, "exactly 50"):
            HallmarkPathwayEncoder([2] * 49, output_dim=4)

        encoder = HallmarkPathwayEncoder([2] * 50, output_dim=4, dropout=0.0)
        pathways = [torch.randn(2) for _ in range(50)]
        pathways[17] = torch.randn(3)
        with self.assertRaisesRegex(ValueError, "Pathway 17 width 3 != 2"):
            encoder(pathways)


class PlainEntropyKLUOTTests(unittest.TestCase):
    def test_one_by_one_solution_matches_plain_entropy_analytic_mass(self):
        cost_value = 0.42
        epsilon, tau_source, tau_target = 0.2, 0.4, 0.7
        cost = torch.tensor([[cost_value]], dtype=torch.float64)
        transport = plain_entropy_kl_uot(
            cost,
            torch.ones(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            epsilon=epsilon,
            tau_source=tau_source,
            tau_target=tau_target,
            iterations=100,
        )
        expected = math.exp(
            -cost_value / (epsilon + tau_source + tau_target)
        )
        self.assertAlmostEqual(transport.item(), expected, places=13)

    def test_general_solution_satisfies_kkt_and_is_differentiable(self):
        cost = torch.tensor(
            [
                [0.20, 0.70, 0.10, 0.50],
                [0.80, 0.30, 0.60, 0.20],
                [0.40, 0.90, 0.25, 0.75],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        source = torch.tensor(
            [0.2, 0.3, 0.5], dtype=torch.float64, requires_grad=True
        )
        target = torch.tensor(
            [0.1, 0.2, 0.3, 0.4], dtype=torch.float64, requires_grad=True
        )
        epsilon, tau_source, tau_target = 0.11, 0.6, 0.8
        transport = plain_entropy_kl_uot(
            cost,
            source,
            target,
            epsilon=epsilon,
            tau_source=tau_source,
            tau_target=tau_target,
            iterations=150,
        )

        row_mass = transport.sum(dim=1)
        column_mass = transport.sum(dim=0)
        kkt_residual = (
            cost
            + epsilon * torch.log(transport)
            + tau_source * torch.log(row_mass / source).unsqueeze(1)
            + tau_target * torch.log(column_mass / target).unsqueeze(0)
        )
        self.assertLess(kkt_residual.detach().abs().max().item(), 1e-10)

        weights = torch.arange(1, 13, dtype=torch.float64).reshape(3, 4)
        (transport * weights).sum().backward()
        _assert_finite_nonzero(self, cost.grad, "cost")
        _assert_finite_nonzero(self, source.grad, "source reference")
        _assert_finite_nonzero(self, target.grad, "target reference")

    def test_half_precision_input_keeps_positive_transport_in_solver_precision(self):
        cost = torch.tensor(
            [[0.0, 2.0], [2.0, 0.0]],
            dtype=torch.float16,
        )
        reference = torch.tensor([0.5, 0.5], dtype=torch.float16)
        transport = plain_entropy_kl_uot(cost, reference, reference)

        self.assertEqual(transport.dtype, torch.float32)
        self.assertTrue(bool((transport > 0).all()))
        self.assertLess(float(transport.min()), torch.finfo(torch.float16).smallest_normal)


class TransportedContextTests(unittest.TestCase):
    def test_context_is_exact_raw_transport_product_and_scales_with_mass(self):
        transport = torch.tensor(
            [[0.2, 0.7], [1.3, 0.4], [0.5, 1.1]], dtype=torch.float64
        )
        values = torch.tensor(
            [[1.0, -2.0, 0.5], [0.3, 0.8, -1.4], [2.2, 0.1, 0.9]],
            dtype=torch.float64,
        )
        context = transported_molecular_context(transport, values)
        expected = transport.transpose(0, 1) @ values
        torch.testing.assert_close(context, expected, rtol=0.0, atol=0.0)

        scale = 3.7
        scaled = transported_molecular_context(scale * transport, values)
        torch.testing.assert_close(scaled, scale * context, rtol=1e-14, atol=1e-14)


class CoarseConditioningTests(unittest.TestCase):
    def test_sparse_original_prototype_ids_and_zero_context_identity(self):
        torch.manual_seed(11)
        model = _make_conditioner()
        centroids = model.morphology_centroids.detach()
        prototype_ids = [2, 15, 7, 15]
        features, _, tokens, occupancy, valid = _sparse_slide(
            centroids, prototype_ids
        )

        self.assertIsNone(model.context_to_patch.bias)
        projected_zero = model.context_to_patch(
            torch.zeros(4, model.context_dim)
        )
        torch.testing.assert_close(
            projected_zero, torch.zeros_like(projected_zero), rtol=0.0, atol=0.0
        )

        random_pathways = torch.randn(50, model.pathway_dim)
        _, diagnostics = model.condition_patch_features(
            patch_features=features,
            pathway_tokens=random_pathways,
            morphology_tokens=tokens,
            morphology_occupancy=occupancy,
            morphology_valid=valid,
        )
        torch.testing.assert_close(
            diagnostics["active_morphology_ids"], torch.tensor([2, 7, 15])
        )
        torch.testing.assert_close(
            diagnostics["sampled_morphology_ids"],
            torch.tensor(prototype_ids),
        )
        inactive = ~valid
        torch.testing.assert_close(
            diagnostics["prototype_molecular_context"][inactive],
            torch.zeros_like(diagnostics["prototype_molecular_context"][inactive]),
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(
            diagnostics["prototype_molecular_context"][valid].abs().sum().item(),
            0.0,
        )

        zero_pathways = torch.zeros(50, model.pathway_dim)
        zero_conditioned, zero_diagnostics = model.condition_patch_features(
            patch_features=features,
            pathway_tokens=zero_pathways,
            morphology_tokens=tokens,
            morphology_occupancy=occupancy,
            morphology_valid=valid,
        )
        torch.testing.assert_close(
            zero_diagnostics["prototype_molecular_context"],
            torch.zeros_like(zero_diagnostics["prototype_molecular_context"]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            zero_conditioned,
            features,
            rtol=0.0,
            atol=0.0,
        )





if __name__ == "__main__":
    unittest.main()
