"""Disentangled and Interpretable Multimodal Attention Fusion (DIMAF).

Framework-compatible reproduction of Eijpe et al. (MICCAI 2025), adapted from
https://github.com/Trustworthy-AI-UU-NKI/DIMAF (AGPL-3.0). The released
computation graph is preserved while validation, device handling, and loss
composition are adapted to MMP's unified survival trainer.

Inputs are 16 PANTHER ``[mixture weight, mixture mean]`` histology tokens and
50 fold-standardized Hallmark RNA pathway tensors. The shared trainer owns the
primary NLL or Cox loss; this module declares DIMAF's distance-correlation objective
through the standard auxiliary-loss contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


class SNNBlock(nn.Module):
    """Official Linear -> ELU -> AlphaDropout pathway block."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.25) -> None:
        super().__init__()
        if int(in_dim) <= 0 or int(out_dim) <= 0:
            raise ValueError("SNN dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(out_dim)),
            nn.ELU(),
            nn.AlphaDropout(p=float(dropout)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class MultiSNN(nn.Module):
    """One independent two-block SNN for every Hallmark pathway."""

    def __init__(
        self,
        in_dims: Sequence[int],
        out_dim: int,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_dims = tuple(int(value) for value in in_dims)
        if not self.in_dims or any(value <= 0 for value in self.in_dims):
            raise ValueError("Every DIMAF RNA pathway must contain at least one gene")
        self.ensemble_snn = nn.ModuleList(
            nn.Sequential(
                SNNBlock(input_dim, out_dim, dropout),
                SNNBlock(out_dim, out_dim, dropout),
            )
            for input_dim in self.in_dims
        )

    def forward(self, pathways: Sequence[torch.Tensor]) -> torch.Tensor:
        if not isinstance(pathways, (list, tuple)):
            raise TypeError("DIMAF RNA input must be a list/tuple of pathway tensors")
        if len(pathways) != len(self.ensemble_snn):
            raise ValueError(
                f"DIMAF received {len(pathways)} RNA pathways, "
                f"expected {len(self.ensemble_snn)}"
            )
        outputs = []
        batch_size: Optional[int] = None
        for index, (network, values, expected_dim) in enumerate(
            zip(self.ensemble_snn, pathways, self.in_dims)
        ):
            if not torch.is_tensor(values):
                values = torch.as_tensor(values)
            if values.ndim == 1:
                values = values.unsqueeze(0)
            if values.ndim != 2 or int(values.shape[1]) != expected_dim:
                raise ValueError(
                    f"DIMAF pathway {index} must be [batch,{expected_dim}], "
                    f"got {tuple(values.shape)}"
                )
            if batch_size is None:
                batch_size = int(values.shape[0])
            elif int(values.shape[0]) != batch_size:
                raise ValueError("DIMAF RNA pathways have inconsistent batch sizes")
            outputs.append(network(values.float()))
        return torch.stack(outputs, dim=1)


class CrossAttentionLayer(nn.Module):
    """Pre-normalized Q/K/V attention used by all four DIMAF paths."""

    def __init__(self, dim: int, dim_head: int, heads: int = 1) -> None:
        super().__init__()
        if min(int(dim), int(dim_head), int(heads)) <= 0:
            raise ValueError("Attention dimensions and heads must be positive")
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.inner_dim = self.heads * self.dim_head
        self.scale = self.dim_head ** -0.5
        self.norm_x = nn.LayerNorm(int(dim))
        self.norm_y = nn.LayerNorm(int(dim))
        self.to_q = nn.Linear(int(dim), self.inner_dim, bias=False)
        self.to_k = nn.Linear(int(dim), self.inner_dim, bias=False)
        self.to_v = nn.Linear(int(dim), self.inner_dim, bias=False)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = values.shape
        return values.reshape(
            batch, tokens, self.heads, self.dim_head
        ).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_attention: bool = False,
    ):
        if x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0]:
            raise ValueError("DIMAF attention inputs must be [batch,tokens,dim]")
        query = self._split_heads(self.to_q(self.norm_x(x))) * self.scale
        key = self._split_heads(self.to_k(self.norm_y(y)))
        value = self._split_heads(self.to_v(self.norm_y(y)))
        attention = torch.matmul(query, key.transpose(-1, -2)).softmax(dim=-1)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(
            x.shape[0], x.shape[1], self.inner_dim
        )
        if return_attention:
            public_attention = attention[:, 0] if self.heads == 1 else attention
            return output, public_attention
        return output


class DistanceCorrelation(nn.Module):
    """Distance correlation used for the paper's D1 and D2 objectives."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        if float(epsilon) <= 0.0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)

    @staticmethod
    def _double_center(distances: torch.Tensor) -> torch.Tensor:
        return (
            distances
            - distances.mean(dim=0, keepdim=True)
            - distances.mean(dim=1, keepdim=True)
            + distances.mean()
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("Distance correlation inputs must be [batch,features]")
        if x.shape[0] == 0:
            raise ValueError("Distance correlation requires a non-empty batch")
        if x.shape[0] < 2:
            # A single patient cannot define a between-patient dependence term.
            return (x.sum() + y.sum()) * 0.0
        centered_x = self._double_center(torch.cdist(x, x, p=2))
        centered_y = self._double_center(torch.cdist(y, y, p=2))
        covariance = (centered_x * centered_y).mean().clamp_min(0.0).sqrt()
        variance_x = centered_x.square().mean().clamp_min(0.0).sqrt()
        variance_y = centered_y.square().mean().clamp_min(0.0).sqrt()
        return covariance / torch.sqrt(variance_x * variance_y + self.epsilon)


class DIMAF(nn.Module):
    """Official DIMAF fusion network with independent survival/auxiliary losses."""

    def __init__(
        self,
        rna_dims: Sequence[int],
        histo_dim: int,
        *,
        num_classes: int = 1,
        single_out_dim: int = 256,
        num_proto_wsi: int = 16,
        prototype_embedding_dim: int = 32,
        dropout: float = 0.25,
        disentanglement_weight: float = 7.0,
        d1_weight: float = 0.5,
        d2_weight: float = 0.5,
        distance_correlation_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.rna_dims = tuple(int(value) for value in rna_dims)
        self.histo_dim = int(histo_dim)
        self.num_classes = int(num_classes)
        self.single_out_dim = int(single_out_dim)
        self.nr_wsi_prototypes = int(num_proto_wsi)
        self.nr_rna_prototypes = len(self.rna_dims)
        self.prototype_embedding_dim = int(prototype_embedding_dim)
        self.disentanglement_weight = float(disentanglement_weight)
        self.d1_weight = float(d1_weight)
        self.d2_weight = float(d2_weight)

        if self.num_classes < 1:
            raise ValueError("DIMAF requires at least one survival output")
        if self.histo_dim <= 0 or self.single_out_dim <= 0:
            raise ValueError("DIMAF embedding dimensions must be positive")
        if self.nr_wsi_prototypes != 16:
            raise ValueError("The paper DIMAF protocol requires exactly 16 WSI prototypes")
        if self.nr_rna_prototypes != 50:
            raise ValueError("The paper DIMAF protocol requires exactly 50 Hallmark pathways")
        if self.prototype_embedding_dim <= 0:
            raise ValueError("prototype_embedding_dim must be positive")
        if min(self.disentanglement_weight, self.d1_weight, self.d2_weight) < 0.0:
            raise ValueError("DIMAF loss weights must be non-negative")
        if self.d1_weight + self.d2_weight <= 0.0:
            raise ValueError("At least one disentanglement component must be active")

        self.f_h = nn.Sequential(nn.Linear(self.histo_dim, self.single_out_dim))
        self.f_g = MultiSNN(self.rna_dims, self.single_out_dim, dropout=dropout)

        fusion_dim = self.single_out_dim + self.prototype_embedding_dim
        if fusion_dim % 2:
            raise ValueError("single_out_dim + prototype_embedding_dim must be even")
        attention_dim = fusion_dim // 2
        self.fusion_dim = fusion_dim
        self.attention_dim = attention_dim

        self.wsi_pt_embedding = nn.Parameter(
            torch.randn(1, self.nr_wsi_prototypes, self.prototype_embedding_dim)
        )
        self.rna_pt_embedding = nn.Parameter(
            torch.randn(1, self.nr_rna_prototypes, self.prototype_embedding_dim)
        )
        self.rna_attention = CrossAttentionLayer(fusion_dim, attention_dim)
        self.wsi_attention = CrossAttentionLayer(fusion_dim, attention_dim)
        self.cross_attention_rna_wsi = CrossAttentionLayer(
            fusion_dim, attention_dim
        )
        self.cross_attention_wsi_rna = CrossAttentionLayer(
            fusion_dim, attention_dim
        )
        self.layer_norm = nn.LayerNorm(attention_dim)
        self.f_surv = nn.Linear(4 * attention_dim, self.num_classes, bias=False)
        self.distance_correlation = DistanceCorrelation(
            distance_correlation_epsilon
        )

    def _validate_wsi(self, wsi: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(wsi):
            wsi = torch.as_tensor(wsi)
        if wsi.ndim == 2:
            wsi = wsi.unsqueeze(0)
        expected = (self.nr_wsi_prototypes, self.histo_dim)
        if wsi.ndim != 3 or tuple(wsi.shape[1:]) != expected:
            raise ValueError(
                f"DIMAF WSI tokens must be [batch,{expected[0]},{expected[1]}], "
                f"got {tuple(wsi.shape)}"
            )
        return wsi.float()

    def _append_prototype_embeddings(
        self,
        wsi_embedding: torch.Tensor,
        rna_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(wsi_embedding.shape[0])
        if int(rna_embedding.shape[0]) != batch_size:
            raise ValueError("DIMAF WSI and RNA batch sizes do not match")
        z_g = torch.cat(
            [rna_embedding, self.rna_pt_embedding.expand(batch_size, -1, -1)],
            dim=-1,
        )
        z_h = torch.cat(
            [wsi_embedding, self.wsi_pt_embedding.expand(batch_size, -1, -1)],
            dim=-1,
        )
        return z_g, z_h

    def _disentangled_attention(
        self,
        z_h: torch.Tensor,
        z_g: torch.Tensor,
        *,
        return_attn: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        matrices: Dict[str, torch.Tensor] = {}

        def attend(name, layer, query, context):
            if return_attn:
                values, matrix = layer(
                    query, context, return_attention=True
                )
                matrices[name] = matrix
                return values
            return layer(query, context)

        z_gg = attend("self_attn_rna", self.rna_attention, z_g, z_g)
        z_hg = attend(
            "cross_attn_wsi_rna", self.cross_attention_wsi_rna, z_g, z_h
        )
        z_gh = attend(
            "cross_attn_rna_wsi", self.cross_attention_rna_wsi, z_h, z_g
        )
        z_hh = attend("self_attn_wsi", self.wsi_attention, z_h, z_h)
        return torch.cat([z_gg, z_hg, z_gh, z_hh], dim=1), matrices

    def forward_no_loss(
        self,
        wsi: torch.Tensor,
        rna: Sequence[torch.Tensor],
        return_attn: bool = False,
    ) -> Dict[str, Any]:
        wsi_embedding = self.f_h(self._validate_wsi(wsi))
        rna_embedding = self.f_g(rna)
        z_g, z_h = self._append_prototype_embeddings(
            wsi_embedding, rna_embedding
        )
        tokens, matrices = self._disentangled_attention(
            z_h, z_g, return_attn=bool(return_attn)
        )
        tokens = self.layer_norm(tokens)

        rna_count = self.nr_rna_prototypes
        wsi_count = self.nr_wsi_prototypes
        z_gg = tokens[:, :rna_count].mean(dim=1)
        z_hg = tokens[:, rna_count : 2 * rna_count].mean(dim=1)
        z_gh = tokens[
            :, 2 * rna_count : 2 * rna_count + wsi_count
        ].mean(dim=1)
        z_hh = tokens[:, 2 * rna_count + wsi_count :].mean(dim=1)

        disentangled = torch.cat([z_hg, z_gh, z_gg, z_hh], dim=1)
        logits = self.f_surv(disentangled)
        d1 = self.distance_correlation(z_gg, z_hh)
        d2 = self.distance_correlation(
            torch.cat([z_hg, z_gh], dim=1),
            torch.cat([z_gg, z_hh], dim=1),
        )

        results: Dict[str, Any] = {
            "logits": logits,
            "wsi_rna_repr": z_hg,
            "rna_wsi_repr": z_gh,
            "rna_repr": z_gg,
            "wsi_repr": z_hh,
            "disentangled_embedding": disentangled,
            "distance_correlation_d1": d1,
            "distance_correlation_d2": d2,
            "distance_correlation_estimable": wsi_embedding.shape[0] > 1,
            "aux_loss": {
                "terms": {
                    "distance_correlation_d1": {
                        "value": d1,
                        "weight": self.disentanglement_weight * self.d1_weight,
                    },
                    "distance_correlation_d2": {
                        "value": d2,
                        "weight": self.disentanglement_weight * self.d2_weight,
                    },
                },
                "nll_logits": {},
            },
        }
        results.update(matrices)
        return results

    forward_mm_no_loss = forward_no_loss

    def forward(
        self,
        wsi: torch.Tensor,
        rna: Sequence[torch.Tensor],
        *,
        return_attn: bool = False,
        **_: object,
    ) -> Dict[str, Any]:
        return self.forward_no_loss(wsi, rna, return_attn=return_attn)


__all__ = [
    "CrossAttentionLayer",
    "DIMAF",
    "DistanceCorrelation",
    "MultiSNN",
    "SNNBlock",
]
