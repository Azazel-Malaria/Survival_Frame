from pathlib import Path
import sys
import unittest

import torch


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mil_models.modal_starpath import (
    GatedSlideAttentionAggregator,
    MeanSlideAggregator,
    build_slide_aggregator,
)


class SlideAggregationTests(unittest.TestCase):
    def test_mean_is_arithmetic_masked_mean_and_permutation_invariant(self):
        features = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 6.0], [999.0, 999.0]],
                [[5.0, 7.0], [9.0, 11.0], [13.0, 17.0]],
            ]
        )
        mask = torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        )
        aggregator = MeanSlideAggregator(input_dim=2)
        expected = torch.tensor([[2.0, 4.0], [9.0, 35.0 / 3.0]])
        actual = aggregator(features, mask)
        torch.testing.assert_close(actual, expected)

        permutation = torch.tensor([2, 0, 1])
        permuted = aggregator(features[:, permutation], mask[:, permutation])
        torch.testing.assert_close(permuted, actual)

    def test_single_slide_is_exact_identity_for_mean_and_gated(self):
        features = torch.tensor(
            [
                [[100.0, 200.0], [2.5, -7.0], [-3.0, 8.0]],
                [[4.0, 5.0], [900.0, 901.0], [800.0, 801.0]],
            ]
        )
        mask = torch.tensor(
            [[False, True, False], [True, False, False]], dtype=torch.bool
        )
        expected = torch.tensor([[2.5, -7.0], [4.0, 5.0]])

        mean = MeanSlideAggregator(input_dim=2)
        gated = GatedSlideAttentionAggregator(input_dim=2, hidden_dim=3)
        torch.testing.assert_close(mean(features, mask), expected)
        torch.testing.assert_close(gated(features, mask), expected)

    def test_gated_attention_remains_an_explicit_opt_in(self):
        mean = build_slide_aggregator("mean", input_dim=2)
        gated = build_slide_aggregator("gated", input_dim=2, hidden_dim=3)
        self.assertIsInstance(mean, MeanSlideAggregator)
        self.assertIsInstance(gated, GatedSlideAttentionAggregator)

        features = torch.tensor(
            [[[1.0, 4.0], [2.0, 3.0], [5.0, -1.0]]]
        )
        mask = torch.ones(1, 3, dtype=torch.bool)
        permutation = torch.tensor([1, 2, 0])
        torch.testing.assert_close(
            gated(features, mask),
            gated(features[:, permutation], mask[:, permutation]),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
