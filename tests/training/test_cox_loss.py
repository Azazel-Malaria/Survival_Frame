from pathlib import Path
import sys
import unittest

import torch


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.losses import CoxLoss, partial_ll_loss
from mil_models.components import process_surv


class CoxLossTests(unittest.TestCase):
    def setUp(self):
        self.log_risks = torch.log(
            torch.tensor([2.0, 3.0, 11.0, 5.0, 7.0], dtype=torch.float64)
        )
        self.times = torch.tensor(
            [5.0, 5.0, 5.0, 3.0, 7.0], dtype=torch.float64
        )
        # At time 5 there are two events and one censoring observation. The
        # censored patient at the tied time must remain in the denominator.
        self.censorships = torch.tensor(
            [0.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float64
        )

    def test_breslow_ties_match_manual_partial_likelihood(self):
        actual = CoxLoss()(
            logits=self.log_risks,
            times=self.times,
            censorships=self.censorships,
        )["loss"]

        # t=5 risk set: 2 + 3 + 11 + 7 = 23, with two tied events.
        # t=3 risk set: 2 + 3 + 11 + 5 + 7 = 28.
        expected_log_likelihood = (
            torch.log(torch.tensor(2.0, dtype=torch.float64))
            + torch.log(torch.tensor(3.0, dtype=torch.float64))
            - 2.0 * torch.log(torch.tensor(23.0, dtype=torch.float64))
            + torch.log(torch.tensor(5.0, dtype=torch.float64))
            - torch.log(torch.tensor(28.0, dtype=torch.float64))
        )
        expected = -expected_log_likelihood / 3.0
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_loss_and_gradients_are_invariant_to_patient_permutation(self):
        base_logits = self.log_risks.clone().requires_grad_(True)
        base_loss = CoxLoss()(
            base_logits, self.times, self.censorships
        )["loss"]
        base_loss.backward()

        permutation = torch.tensor([3, 2, 4, 0, 1])
        permuted_logits = self.log_risks[permutation].clone().requires_grad_(True)
        permuted_loss = CoxLoss()(
            permuted_logits,
            self.times[permutation],
            self.censorships[permutation],
        )["loss"]
        permuted_loss.backward()

        inverse = torch.argsort(permutation)
        torch.testing.assert_close(permuted_loss, base_loss, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            permuted_logits.grad[inverse],
            base_logits.grad,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_all_censored_risk_set_is_differentiable_zero(self):
        logits = torch.tensor(
            [-3.0, 0.5, 8.0], dtype=torch.float64, requires_grad=True
        )
        loss = partial_ll_loss(
            lrisks=logits,
            survival_times=torch.tensor([1.0, 2.0, 3.0]),
            event_indicators=torch.zeros(3),
        )["loss"]

        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_process_surv_uses_continuous_time_not_discrete_bins(self):
        logits = self.log_risks.clone().requires_grad_(True)
        results, _ = process_surv(
            logits,
            torch.zeros_like(self.times),
            self.censorships,
            CoxLoss(),
            survival_time=self.times,
        )
        expected = CoxLoss()(
            logits=logits,
            times=self.times,
            censorships=self.censorships,
        )["loss"]

        torch.testing.assert_close(results["loss"], expected)
        torch.testing.assert_close(results["risk"], logits)


if __name__ == "__main__":
    unittest.main()
