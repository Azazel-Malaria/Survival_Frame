"""TITAN survival baseline used by the unified MMP benchmark.

The official TITAN encoder is frozen and run once by
``training.prepare_titan_embeddings``. This module intentionally contains
only the downstream linear head: importing a survival model must not download
weights, construct CONCH, or move a model to CUDA as a side effect.
"""

from __future__ import annotations

from typing import MutableMapping, Optional

import torch
from torch import nn


class TITANSurvivalHead(nn.Module):
    """Linear survival head over frozen official TITAN slide embeddings.

    TITAN's downstream protocol averages all slide embeddings belonging to a
    patient. ``n_classes=4`` implements the benchmark's four-bin NLL protocol;
    ``n_classes=1`` is also supported for a Cox head without changing the
    encoder or patient aggregation.
    """

    def __init__(
        self,
        input_dim: int = 768,
        n_classes: int = 4,
        *,
        precomputed: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_classes = int(n_classes)
        if not precomputed:
            raise ValueError(
                "TITANSurvivalHead requires embeddings precomputed with the "
                "official local TITAN encoder"
            )
        if self.input_dim != 768:
            raise ValueError(
                "Official MahmoodLab/TITAN slide embeddings have dimension 768"
            )
        if self.n_classes <= 0:
            raise ValueError("n_classes must be positive")

        self.classifier = nn.Linear(self.input_dim, self.n_classes)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        titan_embedding: torch.Tensor,
        *,
        slide_mask: Optional[torch.Tensor] = None,
        results_dict: Optional[MutableMapping[str, object]] = None,
        **_: object,
    ):
        """Aggregate ``[slides,768]`` and emit patient-level raw logits."""

        embedding = torch.as_tensor(titan_embedding)
        if embedding.ndim == 2:
            embedding = embedding.unsqueeze(0)
        if embedding.ndim != 3:
            raise ValueError(
                "TITAN embeddings must be [slides,768] or [batch,slides,768]"
            )
        if embedding.shape[1] == 0 or embedding.shape[2] != self.input_dim:
            raise ValueError(
                f"TITAN slide embeddings must be [batch,slides,{self.input_dim}], "
                f"got {tuple(embedding.shape)}"
            )
        if not torch.is_floating_point(embedding):
            embedding = embedding.float()
        if not torch.isfinite(embedding).all():
            raise ValueError("TITAN slide embeddings contain NaN or Inf")

        if slide_mask is None:
            patient_embedding = embedding.mean(dim=1)
        else:
            mask = torch.as_tensor(
                slide_mask,
                device=embedding.device,
                dtype=embedding.dtype,
            )
            if mask.shape != embedding.shape[:2]:
                raise ValueError(
                    "slide_mask must be [batch,slides] and match TITAN embeddings"
                )
            if not torch.isfinite(mask).all() or not torch.all(
                torch.logical_or(mask == 0, mask == 1)
            ):
                raise ValueError("slide_mask must be finite and binary")
            denominator = mask.sum(dim=1, keepdim=True)
            if torch.any(denominator <= 0):
                raise ValueError("slide_mask must retain at least one slide per patient")
            patient_embedding = (
                embedding * mask.unsqueeze(-1)
            ).sum(dim=1) / denominator

        logits = self.classifier(patient_embedding)
        results = dict(results_dict or {})
        results.update(
            {
                "slide_feat": embedding,
                "patient_feat": patient_embedding,
                "aux_loss": {"terms": {}, "nll_logits": {}},
            }
        )
        if self.n_classes == 1:
            y_hat = None
            y_prob = torch.sigmoid(logits)
        else:
            y_hat = torch.argmax(logits, dim=1)
            y_prob = torch.softmax(logits, dim=1)
        return logits, y_hat, y_prob, results


__all__ = ["TITANSurvivalHead"]
