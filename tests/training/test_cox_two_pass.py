import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.trainer import _deferred_cox_backward, device
from utils.losses import CoxLoss


class _TinyUnifiedCoxModel(nn.Module):
    is_unified_adapter = True

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=True, dtype=torch.float64)

    def forward(self, data, omics, batch=None, **kwargs):
        del omics, batch, kwargs
        logits = self.linear(data)
        return {
            "logits": logits,
            "risk": logits,
            "auxiliary_loss": None,
        }, {}


def _patient_batch(features, time, censorship):
    return {
        "img": features.reshape(1, -1),
        "survival_time": torch.tensor([time], dtype=torch.float64),
        "censorship": torch.tensor([censorship], dtype=torch.float64),
        "label": torch.tensor([0], dtype=torch.long),
    }


class DeferredCoxBackwardTests(unittest.TestCase):
    def test_two_pass_vjp_matches_full_risk_set_gradient(self):
        torch.manual_seed(19)
        template = _TinyUnifiedCoxModel()
        deferred_model = copy.deepcopy(template).to(device)
        direct_model = copy.deepcopy(template).to(device)

        features = torch.tensor(
            [
                [1.0, -2.0, 0.5],
                [0.2, 1.5, -1.0],
                [-0.7, 0.4, 2.0],
                [1.2, 0.1, -0.3],
            ],
            dtype=torch.float64,
        )
        times = torch.tensor([9.0, 6.0, 6.0, 2.0], dtype=torch.float64)
        censorships = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        accum_steps = 2

        direct_logits = direct_model(features.to(device), None)
        # The direct model normally returns the adapter-style tuple.
        direct_logits = direct_logits[0]["logits"]
        direct_loss = CoxLoss()(
            direct_logits,
            times.to(device),
            censorships.to(device),
        )["loss"] / float(accum_steps)
        direct_loss.backward()

        patient_batches = [
            _patient_batch(features[index], times[index], censorships[index])
            for index in range(len(features))
        ]
        result = _deferred_cox_backward(
            deferred_model,
            patient_batches,
            CoxLoss(),
            accum_steps=accum_steps,
            expected_riskset_size=len(patient_batches),
        )

        for deferred_parameter, direct_parameter in zip(
            deferred_model.parameters(), direct_model.parameters()
        ):
            torch.testing.assert_close(
                deferred_parameter.grad,
                direct_parameter.grad,
                rtol=1e-10,
                atol=1e-10,
            )
        self.assertEqual(result["patient_count"], len(patient_batches))
        self.assertEqual(tuple(result["risk"].shape), (len(patient_batches), 1))
        self.assertAlmostEqual(
            result["log_dict"]["surv_loss"],
            direct_loss.item() * accum_steps,
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
