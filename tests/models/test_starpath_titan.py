from pathlib import Path
import sys
import unittest

import torch
from torch import nn

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from mil_models.TITAN_STARPath.vision_transformer import CustomSequential, VisionTransformer


class AddOne(nn.Module):
    def forward(self, hidden, attn_mask, bg_mask):
        return hidden + 1


class TitanCallbackTests(unittest.TestCase):
    def test_preblock_injection_preserves_cls_and_postblock_observation(self):
        blocks = CustomSequential(AddOne(), AddOne())
        observed = []
        callbacks = []

        def inject(layer, hidden, auxiliary):
            callbacks.append((layer, hidden.clone()))
            return torch.ones_like(hidden) * 3

        def observe(layer, hidden, auxiliary):
            observed.append((layer, hidden.clone()))

        result = blocks(torch.zeros(1, 3, 4), None, inject_layers=[1],
                        inject_callback=inject, num_prefix_tokens=1,
                        post_block_callback=observe)
        self.assertEqual([x[0] for x in callbacks], [1])
        self.assertEqual([x[0] for x in observed], [0, 1])
        torch.testing.assert_close(callbacks[0][1], torch.ones(1, 3, 4))
        torch.testing.assert_close(result[:, :1], torch.full((1, 1, 4), 2.0))
        torch.testing.assert_close(result[:, 1:], torch.full((1, 2, 4), 5.0))

    def test_callback_auxiliary_follows_sparse_grid_order_and_background_mask(self):
        torch.manual_seed(29)
        model = VisionTransformer(embed_dim=8, mlp_patch_embed_dim=8, depth=2,
                                  num_heads=2, global_pool="avg").eval()
        features = torch.randn(1, 3, 8)
        coords = torch.tensor([[[512, 0], [0, 512], [0, 0]]])
        auxiliary = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        observed = []

        def inject(layer, hidden, aux):
            observed.append(aux.clone())
            return torch.zeros_like(hidden)

        with torch.no_grad():
            output = model(features, coords, 512, inject_layers=[1],
                           inject_callback=inject, inject_callback_context=auxiliary)
        expected = torch.cat((torch.zeros(1, 1, 2), auxiliary[:, [2, 1, 0]]), dim=1)
        torch.testing.assert_close(observed[0], expected)
        self.assertTrue(torch.isfinite(output).all())

    def test_callback_requires_explicit_layers_and_observer_cannot_replace_hidden(self):
        blocks = CustomSequential(AddOne())
        hidden = torch.zeros(1, 2, 4)
        with self.assertRaisesRegex(ValueError, "explicit"):
            blocks(hidden, None, inject_callback=lambda *_: None)
        with self.assertRaisesRegex(TypeError, "must return None"):
            blocks(hidden, None, post_block_callback=lambda *_: hidden)
        with self.assertRaises(TypeError):
            blocks(hidden, None, inject_context=hidden)


if __name__ == "__main__":
    unittest.main()
