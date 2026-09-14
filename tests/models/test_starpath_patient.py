from pathlib import Path
import sys
import unittest

import torch
from torch import nn

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from mil_models.modal_starpath import STARPathPatientAdapter, MeanSlideAggregator


class RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.st_gene_seq = ("ST0", "ST1", "ST2")
        self.classifier = nn.Linear(768, 2)
        self.calls = []
        self.rna_calls = 0

    def encode_coarse_rna_pathways(self, pathways):
        self.rna_calls += 1
        return torch.stack([values.mean() for values in pathways])

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return None, None, None, {
            "slide_feat": kwargs["x1"].mean(dim=0, keepdim=True),
            "aux_loss": {"terms": {}, "nll_logits": {}},
        }


def payload():
    count = torch.zeros(16, dtype=torch.long)
    count[3] = 4
    atlas = torch.zeros(16, 3)
    atlas[3] = torch.tensor([0.2, 0.3, 0.4])
    return {
        "slides": [torch.ones(4, 768), torch.ones(6, 768) * 3],
        "coords": [torch.zeros(4, 2), torch.zeros(6, 2)],
        "st": [torch.ones(4, 3), torch.ones(6, 3)],
        "morphology_tokens": [torch.zeros(16, 768)] * 2,
        "morphology_occupancy": [(count > 0).float()] * 2,
        "morphology_valid": [count > 0] * 2,
        "route_atlas": [atlas.clone(), atlas.clone()],
        "route_atlas_counts": [count.clone(), count.clone()],
        "route_atlas_valid": [count > 0, count > 0],
        "slide_ids": ["slide-a", "slide-b"],
        "patch_sizes": [512, 512],
    }


def omics():
    return {
        "coarse_pathways": [torch.tensor([float(i)]) for i in range(50)],
        "fine_rna": torch.zeros(3),
    }


class PatientAdapterTests(unittest.TestCase):
    def test_rna_is_encoded_once_and_slides_are_averaged_per_patient(self):
        backbone = RecordingBackbone()
        adapter = STARPathPatientAdapter(backbone, MeanSlideAggregator())
        data, rna = payload(), omics()
        results, _ = adapter(data, rna)
        self.assertEqual(backbone.rna_calls, 1)
        self.assertEqual(len(backbone.calls), 2)
        self.assertIs(backbone.calls[0]["coarse_pathway_tokens"], backbone.calls[1]["coarse_pathway_tokens"])
        for i, call in enumerate(backbone.calls):
            self.assertIs(call["rna"], rna["fine_rna"])
            self.assertIs(call["route_atlas"], data["route_atlas"][i])
        torch.testing.assert_close(results["patient_feat"], torch.full((1, 768), 2.0))
        self.assertEqual(results["slide_feat"].shape, (1, 2, 768))

    def test_complete_atlas_and_canonical_field_names_are_required(self):
        for field in ("route_atlas", "route_atlas_counts", "route_atlas_valid", "slides"):
            with self.subTest(field=field):
                backbone = RecordingBackbone()
                adapter = STARPathPatientAdapter(backbone, MeanSlideAggregator())
                data = payload()
                data["features"] = data.pop(field)
                with self.assertRaisesRegex(KeyError, field):
                    adapter(data, omics())
                self.assertEqual(backbone.calls, [])

    def test_inconsistent_atlas_metadata_is_rejected(self):
        for case in ("width", "counts", "validity", "empty"):
            with self.subTest(case=case):
                backbone = RecordingBackbone()
                adapter = STARPathPatientAdapter(backbone, MeanSlideAggregator())
                data = payload()
                if case == "width":
                    data["route_atlas"][0] = torch.zeros(16, 2)
                elif case == "counts":
                    data["route_atlas_counts"][0] = torch.zeros(15)
                elif case == "validity":
                    data["route_atlas_valid"][0][3] = False
                else:
                    data["route_atlas"][0][0, 0] = 1
                with self.assertRaises(ValueError):
                    adapter(data, omics())
                self.assertEqual(backbone.calls, [])


if __name__ == "__main__":
    unittest.main()
