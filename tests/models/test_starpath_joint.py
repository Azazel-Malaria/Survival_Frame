from pathlib import Path
from types import SimpleNamespace
import importlib
import sys
import unittest
from unittest import mock

import torch
from torch import nn
import torch.nn.functional as F


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

starpath_module = importlib.import_module("mil_models.modal_starpath")
STARPath = starpath_module.STARPath


def _assert_module_received_gradient(test_case, module, name):
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    test_case.assertTrue(gradients, f"{name} received no gradients")
    test_case.assertTrue(
        all(torch.isfinite(gradient).all() for gradient in gradients),
        f"{name} received a non-finite gradient",
    )
    test_case.assertGreater(
        sum(gradient.abs().sum().item() for gradient in gradients),
        0.0,
        f"{name} gradients are all zero",
    )


class _FakeTitanBlock(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        base = torch.linspace(-0.05, 0.05, 768) * float(layer_id + 1)
        self.offset = nn.Parameter(base)

    def forward(self, hidden):
        return hidden + self.offset.reshape(1, 1, -1)


class _FakeTitan(nn.Module):
    """One-stream callback-aware TITAN substitute with six real parameters."""

    def __init__(self):
        super().__init__()
        blocks = nn.ModuleList(_FakeTitanBlock(index) for index in range(6))
        self.vision_encoder = nn.Module()
        self.vision_encoder.blocks = nn.Module()
        self.vision_encoder.blocks.modules_list = blocks
        self.vision_encoder.proj = nn.Parameter(torch.eye(768))
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(depth=6)
        )
        self.calls = 0
        self.events = []
        self.last_input = None
        self.last_raw_embedding = None
        self.last_post_block_callback = None

    def encode_slide_from_patch_features(
        self,
        patch_features,
        patch_coords,
        patch_size_lv0,
        *,
        inject_layers=None,
        inject_callback=None,
        inject_callback_context=None,
        post_block_callback=None,
    ):
        self.calls += 1
        self.last_input = patch_features.detach().clone()
        self.last_post_block_callback = post_block_callback
        if tuple(patch_coords.shape) != (1, patch_features.shape[1], 2):
            raise ValueError("fake TITAN received misaligned coordinates")
        if int(patch_size_lv0) <= 0:
            raise ValueError("fake TITAN requires a positive patch size")

        hidden = torch.cat(
            (patch_features.new_zeros(1, 1, 768), patch_features), dim=1
        )
        auxiliary = None
        if inject_callback_context is not None:
            auxiliary = torch.cat(
                (
                    inject_callback_context.new_zeros(
                        1, 1, inject_callback_context.shape[-1]
                    ),
                    inject_callback_context,
                ),
                dim=1,
            )
        active_layers = set(inject_layers or ())
        for layer_id, block in enumerate(
            self.vision_encoder.blocks.modules_list
        ):
            if inject_callback is not None and layer_id in active_layers:
                self.events.append(f"I{layer_id}")
                delta = inject_callback(layer_id, hidden, auxiliary)
                if delta is not None:
                    hidden = hidden + delta
            hidden = block(hidden)
            if post_block_callback is not None:
                post_block_callback(layer_id, hidden, auxiliary)
        embedding = hidden[:, 1:, :].mean(dim=1)
        self.last_raw_embedding = embedding.detach().clone()
        return embedding


def _build_joint_model(**overrides):
    centroids = torch.full((16, 768), 100.0)
    centroids[0].zero_()
    kwargs = dict(
        input_dim=768,
        n_classes=2,
        omic_sizes=[1] * 50,
        pathway_names=[f"HALLMARK_{index:02d}" for index in range(50)],
        morphology_centroids=centroids,
        rna_gene_seq=("JUNB",),
        st_gene_seq=("JUNB",),
        pathway_signature_path=str(
            SRC / "data_csvs" / "rna" / "metadata" / "hallmarks_signatures.csv"
        ),
        model_path=str(SRC / "mil_models" / "TITAN_STARPath"),
        pathway_dim=8,
        pathway_hidden_dim=8,
        pathway_dropout=0.0,
        region_num=2,
        region_assignment_dim=8,
        region_seed_knn=2,
        region_candidate_topk=2,
        transport_sinkhorn_iters=5,
        coarse_pathway_dim=8,
        coarse_compatibility_dim=8,
        coarse_context_dim=8,
        coarse_pathway_dropout=0.0,
        coarse_uot_iterations=5,
    )
    kwargs.update(overrides)
    fake_titan = _FakeTitan()
    with mock.patch.object(
        starpath_module,
        "_load_titan",
        return_value=fake_titan,
    ):
        model = STARPath(**kwargs)
    return model, fake_titan, centroids


def _route_atlas_payload():
    atlas = torch.zeros(16, 1)
    atlas[0, 0] = 0.1
    atlas[1, 0] = 1.0
    counts = torch.zeros(16, dtype=torch.long)
    counts[:2] = 20
    return {
        "route_atlas": atlas,
        "route_atlas_counts": counts,
        "route_atlas_valid": counts > 0,
    }


def _forward_joint_slide(
    model, centroids, *, include_atlas=True, atlas_payload=None
):
    raw_x = 0.03 * torch.randn(4, 768)
    coords = torch.tensor(
        [[0, 0], [512, 0], [0, 512], [512, 512]], dtype=torch.int64
    )
    st = torch.tensor([[0.2], [0.5], [0.8], [1.1]])
    coarse_pathways = [
        torch.tensor([0.1 + index / 100.0]) for index in range(50)
    ]
    coarse_tokens = model.encode_coarse_rna_pathways(coarse_pathways)
    morphology_occupancy = torch.zeros(16)
    morphology_occupancy[0] = 1.0
    atlas_kwargs = {}
    if include_atlas:
        atlas_kwargs = (
            _route_atlas_payload() if atlas_payload is None else atlas_payload
        )
    return model(
        x1=raw_x,
        coords=coords,
        st=st,
        rna=torch.tensor([0.7]),
        coarse_pathway_tokens=coarse_tokens,
        morphology_tokens=centroids.clone(),
        morphology_occupancy=morphology_occupancy,
        morphology_valid=morphology_occupancy > 0,
        patch_size_lv0=512,
        return_slide_features_only=True,
        **atlas_kwargs,
    )[3]


class JointCoarseFineIntegrationTests(unittest.TestCase):
    def test_default_c_single_titan_dynamic_route_gradients(self):
        torch.manual_seed(101)
        model, titan, centroids = _build_joint_model()
        model.train()

        self.assertTrue(titan.training)
        self.assertEqual(model.titan_inject_layers, [2, 4])
        self.assertEqual(model.titan_trainable_layers, [2, 3, 4, 5])
        self.assertFalse(hasattr(model, "state_codebook"))
        self.assertFalse(hasattr(model, "titan_train_mode"))
        blocks = titan.vision_encoder.blocks.modules_list
        for layer_id, block in enumerate(blocks):
            expected = layer_id in {2, 3, 4, 5}
            self.assertTrue(
                all(parameter.requires_grad is expected for parameter in block.parameters())
            )
        self.assertFalse(titan.vision_encoder.proj.requires_grad)

        raw_x = (0.03 * torch.randn(4, 768)).requires_grad_()
        coords = torch.tensor(
            [[0, 0], [512, 0], [0, 512], [512, 512]], dtype=torch.int64
        )
        st = torch.tensor([[0.2], [0.5], [0.8], [1.1]])
        fine_rna = torch.tensor([0.7])
        morphology_tokens = centroids.clone()
        morphology_occupancy = torch.zeros(16)
        morphology_occupancy[0] = 1.0
        morphology_valid = morphology_occupancy > 0
        coarse_pathways = [
            torch.tensor([0.1 + index / 100.0], requires_grad=True)
            for index in range(50)
        ]
        coarse_tokens = model.encode_coarse_rna_pathways(coarse_pathways)

        region_inputs = []

        def capture_region_input(_module, args):
            region_inputs.append(args[0].detach().clone())

        hook = model.region_constructor.register_forward_pre_hook(
            capture_region_input
        )
        try:
            _, _, _, results = model(
                x1=raw_x,
                coords=coords,
                st=st,
                rna=fine_rna,
                coarse_pathway_tokens=coarse_tokens,
                morphology_tokens=morphology_tokens,
                morphology_occupancy=morphology_occupancy,
                morphology_valid=morphology_valid,
                patch_size_lv0=512,
                return_slide_features_only=True,
                **_route_atlas_payload(),
            )
        finally:
            hook.remove()

        self.assertEqual(titan.calls, 1)
        self.assertEqual(titan.events, ["I2", "I4"])
        self.assertIsNotNone(titan.last_post_block_callback)
        self.assertEqual(len(region_inputs), 1)
        torch.testing.assert_close(region_inputs[0], raw_x.detach())
        self.assertFalse(torch.equal(titan.last_input.squeeze(0), raw_x.detach()))
        self.assertEqual(results["fine_region_patch_stream"], "raw")
        self.assertEqual(results["titan_patch_stream"], "coarse_conditioned")
        self.assertEqual(results["titan_inject_layers"], [2, 4])
        self.assertEqual(results["titan_dynamic_write_layer"], 3)
        self.assertGreater(results["transport_atlas_cost_delta"].max().item(), 0.0)
        torch.testing.assert_close(
            results["titan_region_memory_read_l2"],
            results["titan_region_memory_read_l4"],
            rtol=0,
            atol=0,
        )
        self.assertFalse(torch.equal(
            results["titan_region_memory_key_l2"],
            results["titan_region_memory_key_l4"],
        ))
        self.assertNotIn("titan_train_mode", results)

        titan_embedding = F.normalize(
            titan.last_raw_embedding @ titan.vision_encoder.proj,
            dim=-1,
        )
        expected_slide = model.slide_final_norm(titan_embedding)
        torch.testing.assert_close(results["slide_feat"], expected_slide)

        weights = torch.linspace(-1.0, 1.0, 768).reshape(1, -1)
        objective = (results["slide_feat"] * weights).sum()
        objective = objective + 0.05 * (
            results["region_loss"]
            + results["alignment_loss"]
        )
        objective.backward()

        for layer_id in (0, 1):
            self.assertFalse(blocks[layer_id].offset.requires_grad)
            self.assertIsNone(blocks[layer_id].offset.grad)
        for layer_id in (2, 3, 4, 5):
            self.assertTrue(blocks[layer_id].offset.requires_grad)
            self.assertIsNotNone(blocks[layer_id].offset.grad)
            self.assertGreater(blocks[layer_id].offset.grad.abs().sum().item(), 0.0)
        _assert_module_received_gradient(
            self, model.coarse_conditioner, "coarse conditioner"
        )
        _assert_module_received_gradient(
            self, model.pathway_encoder, "fine RNA/ST pathway encoder"
        )
        _assert_module_received_gradient(
            self, model.memory_reader, "fine TITAN memory injector"
        )
        _assert_module_received_gradient(
            self, model.dynamic_memory_writer, "dynamic region-key writer"
        )
        self.assertIsNotNone(raw_x.grad)
        self.assertGreater(raw_x.grad.abs().sum().item(), 0.0)
        for index, pathway in enumerate(coarse_pathways):
            self.assertIsNotNone(pathway.grad, f"coarse pathway {index}")
            self.assertGreater(
                pathway.grad.abs().sum().item(), 0.0, f"coarse pathway {index}"
            )

    def test_custom_trainable_layers_are_independent_of_injection(self):
        torch.manual_seed(202)
        model, titan, centroids = _build_joint_model(
            titan_inject_layers=[0, 2],
            titan_trainable_layers=[0, 1, 2],
        )
        model.train()

        self.assertEqual(model.titan_inject_layers, [0, 2])
        self.assertEqual(model.titan_trainable_layers, [0, 1, 2])
        self.assertEqual(model.titan_frozen_layers, [3, 4, 5])
        blocks = titan.vision_encoder.blocks.modules_list
        for layer_id, block in enumerate(blocks):
            expected = layer_id in {0, 1, 2}
            self.assertTrue(
                all(
                    parameter.requires_grad is expected
                    for parameter in block.parameters()
                )
            )
        self.assertFalse(titan.vision_encoder.proj.requires_grad)

        results = _forward_joint_slide(model, centroids)
        self.assertEqual(titan.events, ["I0", "I2"])
        self.assertEqual(results["titan_inject_layers"], [0, 2])
        self.assertEqual(results["titan_trainable_layers"], [0, 1, 2])
        self.assertIn("titan_inject_delta_l0", results)
        self.assertIn("titan_inject_delta_l2", results)
        self.assertNotIn("titan_inject_delta_l1", results)

    def test_disabling_unfreeze_keeps_all_titan_weights_frozen_and_eval(self):
        torch.manual_seed(303)
        model, titan, centroids = _build_joint_model(
            titan_inject_layers=[1, 4],
            titan_trainable_layers="none",
        )
        model.train()

        self.assertEqual(model.titan_inject_layers, [1, 4])
        self.assertEqual(model.titan_trainable_layers, [])
        self.assertEqual(model.titan_frozen_layers, list(range(6)))
        self.assertFalse(titan.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in titan.parameters())
        )

        results = _forward_joint_slide(model, centroids)
        self.assertEqual(titan.events, ["I1", "I4"])
        self.assertEqual(results["titan_trainable_layers"], [])
        self.assertIn("titan_inject_delta_l1", results)
        self.assertIn("titan_inject_delta_l4", results)


    def test_atlas_flat_and_single_anchor_inputs_are_neutral(self):
        model, _, centroids = _build_joint_model()
        flat = _route_atlas_payload()
        flat["route_atlas"][:2] = 0.5
        flat_result = _forward_joint_slide(
            model, centroids, atlas_payload=flat
        )
        self.assertEqual(
            flat_result["route_atlas_confidence"].count_nonzero(), 0
        )
        self.assertEqual(
            flat_result["transport_atlas_cost_delta"].count_nonzero(), 0
        )

        single = _route_atlas_payload()
        single["route_atlas"][1].zero_()
        single["route_atlas_counts"][1] = 0
        single["route_atlas_valid"] = single["route_atlas_counts"] > 0
        single_result = _forward_joint_slide(
            model, centroids, atlas_payload=single
        )
        self.assertEqual(
            single_result["route_atlas_confidence"].count_nonzero(), 0
        )
        self.assertEqual(
            single_result["transport_atlas_cost_delta"].count_nonzero(), 0
        )



    def test_trainable_layers_can_exclude_an_injected_layer(self):
        model, titan, centroids = _build_joint_model(
            titan_inject_layers=[1, 4],
            titan_trainable_layers=[2, 3],
        )
        self.assertEqual(model.titan_trainable_layers, [2, 3])
        self.assertEqual(model.titan_frozen_layers, [0, 1, 4, 5])
        blocks = titan.vision_encoder.blocks.modules_list
        self.assertTrue(blocks[2].offset.requires_grad)
        self.assertTrue(blocks[3].offset.requires_grad)
        self.assertFalse(blocks[1].offset.requires_grad)
        self.assertFalse(blocks[4].offset.requires_grad)
        results = _forward_joint_slide(model, centroids)
        self.assertEqual(titan.events, ["I1", "I4"])
        self.assertEqual(results["titan_inject_layers"], [1, 4])
        self.assertEqual(results["titan_trainable_layers"], [2, 3])


    def test_single_and_multiple_injection_schedules(self):
        for layers, write_layer in (([2], 1), ([1, 2, 4], 3)):
            with self.subTest(layers=layers):
                torch.manual_seed(411)
                model, titan, centroids = _build_joint_model(titan_inject_layers=layers)
                result = _forward_joint_slide(model, centroids)
                self.assertEqual(titan.events, [f"I{x}" for x in layers])
                self.assertEqual(result["titan_dynamic_write_layer"], write_layer)
                self.assertTrue(torch.isfinite(result["slide_feat"]).all())
                self.assertEqual(result["titan_layer_allocation"].shape, (len(layers),))

    def test_requires_route_atlas_and_rejects_old_options(self):
        model, _, centroids = _build_joint_model()
        with self.assertRaisesRegex(ValueError, "requires route_atlas"):
            _forward_joint_slide(model, centroids, include_atlas=False)
        for kwargs in ({"starpath_variant": "A"}, {"titan_unfreeze_injected_layers": True}, {"titan_package_root": "TITAN"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(TypeError, "Unknown STARPath"):
                _build_joint_model(**kwargs)

    def test_rejects_invalid_layers_and_official_titan_resource_path(self):
        for layers in ([], [0], [-1, 2], [2, 6]):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                _build_joint_model(titan_inject_layers=layers)
        for layers in (True, [False, 2]):
            with self.subTest(layers=layers), self.assertRaises(TypeError):
                _build_joint_model(titan_trainable_layers=layers)
        for layers in ([-1], [6]):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                _build_joint_model(titan_trainable_layers=layers)
        with self.assertRaisesRegex(ValueError, "TITAN_STARPath"):
            _build_joint_model(model_path=str(SRC / "mil_models" / "TITAN"))


if __name__ == "__main__":
    unittest.main()
