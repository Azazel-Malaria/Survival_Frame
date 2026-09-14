from pathlib import Path
import sys
import unittest

import torch


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mil_models.model_titan import TITANSurvivalHead
from wsi_datasets.unified_survival import titan_collate


def _item(embedding, label, time, censorship):
    return {
        "img": embedding,
        "label": torch.tensor(label, dtype=torch.long),
        "survival_time": torch.tensor(time, dtype=torch.float32),
        "censorship": torch.tensor(censorship, dtype=torch.float32),
    }


class TitanCollateTests(unittest.TestCase):
    def test_variable_slide_counts_are_padded_and_masked_mean_is_exact(self):
        first = torch.stack(
            [torch.full((768,), 1.0), torch.full((768,), 3.0)]
        )
        second = torch.stack(
            [
                torch.full((768,), 2.0),
                torch.full((768,), 4.0),
                torch.full((768,), 8.0),
            ]
        )
        batch = titan_collate(
            [
                _item(first, label=0, time=5.0, censorship=0.0),
                _item(second, label=1, time=8.0, censorship=1.0),
            ]
        )

        self.assertEqual(tuple(batch["img"].shape), (2, 3, 768))
        self.assertEqual(batch["slide_mask"].dtype, torch.bool)
        torch.testing.assert_close(
            batch["slide_mask"],
            torch.tensor([[True, True, False], [True, True, True]]),
        )
        torch.testing.assert_close(batch["img"][0, 2], torch.zeros(768))

        # Put an extreme value into the padded slot: the TITAN head must still
        # recover the original per-patient slide means through slide_mask.
        padded = batch["img"].clone()
        padded[~batch["slide_mask"]] = 10000.0
        head = TITANSurvivalHead(n_classes=1)
        _, _, _, results = head(padded, slide_mask=batch["slide_mask"])
        expected = torch.stack(
            [first.mean(dim=0), second.mean(dim=0)], dim=0
        )
        torch.testing.assert_close(results["patient_feat"], expected)

    def test_collate_rejects_non_official_feature_dimension(self):
        bad_item = _item(
            torch.zeros(2, 16), label=0, time=1.0, censorship=1.0
        )
        with self.assertRaisesRegex(ValueError, "dimension must be 768"):
            titan_collate([bad_item])


if __name__ == "__main__":
    unittest.main()
