"""Thin adapters that make heterogeneous survival backbones share MMP's API."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn

from mil_models.components import process_surv
from utils.losses import NLLSurvLoss


def _as_batch_matrix(value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 1:
        return value.unsqueeze(0)
    return value


def _as_single_sample_vector(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dim() == 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.dim() != 1:
        raise ValueError(
            f"{name} must have shape [genes] or [1,genes], got {tuple(value.shape)}"
        )
    return value


def _remove_single_patient_padding(
    data: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    model_name: str,
) -> torch.Tensor:
    """Drop dataset padding before calling a single-patient WSI backbone."""
    if attn_mask is None:
        if data.ndim == 3 and data.shape[0] == 1:
            data = data.squeeze(0)
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError(f"{model_name} expects one nonempty patient WSI bag")
        return data
    if not torch.is_tensor(data) or not torch.is_tensor(attn_mask):
        raise TypeError(f"{model_name} data and attn_mask must be torch.Tensor objects")

    if data.dim() == 3:
        if data.shape[0] != 1:
            raise ValueError(
                f"{model_name} only supports one patient per WSI bag, got {tuple(data.shape)}"
            )
        data = data.squeeze(0)
    if data.dim() != 2:
        raise ValueError(
            f"{model_name} WSI data must have shape [patches,features] or "
            f"[1,patches,features], got {tuple(data.shape)}"
        )

    if attn_mask.dim() == 2:
        if attn_mask.shape[0] != 1:
            raise ValueError(
                f"{model_name} only supports one padding mask, got {tuple(attn_mask.shape)}"
            )
        attn_mask = attn_mask.squeeze(0)
    if attn_mask.dim() != 1 or attn_mask.shape[0] != data.shape[0]:
        raise ValueError(
            f"{model_name} attn_mask must match the patch axis; got "
            f"data={tuple(data.shape)}, mask={tuple(attn_mask.shape)}"
        )
    if attn_mask.is_floating_point() and not bool(torch.isfinite(attn_mask).all().item()):
        raise ValueError(f"{model_name} attn_mask contains non-finite values")
    if not bool(((attn_mask == 0) | (attn_mask == 1)).all().item()):
        raise ValueError(f"{model_name} attn_mask must contain only 0/1 values")

    valid = attn_mask.to(device=data.device, dtype=torch.bool)
    if not bool(valid.any().item()):
        raise ValueError(f"{model_name} attn_mask contains no valid patches")
    return data[valid]


class UnifiedSurvivalAdapter(nn.Module):
    """Map a copied/reference backbone onto the existing trainer contract.

    The adapter owns no training policy.  It only translates arguments, calls
    ``process_surv`` exactly once, and adds explicitly declared auxiliary terms
    during training.  The outer epoch/checkpoint/C-index loop remains shared.
    """

    is_unified_adapter = True

    def __init__(self, backbone: nn.Module, model_name: str, *, aux_nll_alpha=0.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.model_name = str(model_name).lower()
        self.supports_batched_patients = self.model_name not in {
            "abmil", "transmil", "mcat", "survpath", "slotspe"
        }
        self.needs_discrete_labels = self.model_name == "slotspe"
        self.aux_nll_loss = NLLSurvLoss(alpha=aux_nll_alpha)

    def _call_backbone(
        self,
        data: Any,
        omics: Any,
        *,
        return_attn: bool,
        batch: Optional[Mapping[str, Any]],
    ):
        name = self.model_name
        payload: Dict[str, Any] = {}

        if name == "abmil":
            logits = self.backbone(data_WSI=data)
        elif name == "transmil":
            logits = self.backbone(data_WSI=data)
        elif name == "mcat":
            if not isinstance(omics, (list, tuple)) or len(omics) != 6:
                raise ValueError("MCAT requires exactly six functional gene families")
            kwargs = {"x_path": data}
            kwargs.update({
                f"x_omic{idx + 1}": _as_single_sample_vector(
                    value, f"MCAT functional family {idx + 1}"
                )
                for idx, value in enumerate(omics)
            })
            logits = self.backbone(**kwargs)
        elif name in {"mlp", "snn", "s_mlp"}:
            logits = self.backbone(data_omics=_as_batch_matrix(omics))
        elif name == "dimaf":
            payload = self.backbone(data, omics, return_attn=return_attn)
            logits = payload["logits"]
        elif name in {"mmp_trans", "mmp_ot", "survpath"}:
            if data.ndim == 2:
                data = data.unsqueeze(0)
            pathways = [_as_batch_matrix(value) for value in omics]
            payload = self.backbone.forward_no_loss(data, pathways, return_attn=return_attn)
            logits = payload["logits"]
        elif name == "titan":
            slide_mask = batch.get("slide_mask") if batch is not None else None
            logits, _, _, payload = self.backbone(data, slide_mask=slide_mask)
        elif name == "slotspe":
            logits, _, _, payload = self.backbone(
                x1=data,
                rna=None if omics is None else _as_batch_matrix(omics),
                return_attn=return_attn,
            )
        else:
            raise NotImplementedError(f"No unified adapter mapping for {name!r}")
        return logits, payload

    @staticmethod
    def _add_auxiliary_losses(
        results: Dict[str, Any],
        log_dict: Dict[str, float],
        payload: Mapping[str, Any],
        *,
        loss_fn,
        label: torch.Tensor,
        censorship: torch.Tensor,
        enabled: bool,
        discrete_label=None,
        aux_nll_loss=None,
    ) -> None:
        if not enabled:
            return
        aux_spec = payload.get("aux_loss", {}) if payload else {}
        auxiliary_total = None

        for term_name, term in aux_spec.get("terms", {}).items():
            value = term.get("value") if isinstance(term, Mapping) else term
            weight = term.get("weight", 1.0) if isinstance(term, Mapping) else 1.0
            if value is None:
                continue
            if not torch.is_tensor(value):
                value = results["logits"].new_tensor(float(value))
            weighted = value.mean() * float(weight)
            auxiliary_total = weighted if auxiliary_total is None else auxiliary_total + weighted
            log_dict[f"aux_{term_name}"] = float(value.detach().mean().item())
            log_dict[f"weighted_aux_{term_name}"] = float(weighted.detach().item())

        for term_name, term in aux_spec.get("nll_logits", {}).items():
            nll_fn = aux_nll_loss if aux_nll_loss is not None else loss_fn
            if not isinstance(nll_fn, NLLSurvLoss):
                raise TypeError(f"Auxiliary NLL head {term_name!r} requires its own NLLSurvLoss")
            nll_label = discrete_label if discrete_label is not None else (
                label if isinstance(loss_fn, NLLSurvLoss) else None
            )
            if nll_label is None:
                raise ValueError(f"Auxiliary NLL head {term_name!r} requires discrete_label")
            logits = term["logits"] if isinstance(term, Mapping) else term
            weight = term.get("weight", 1.0) if isinstance(term, Mapping) else 1.0
            aux_nll = nll_fn(
                logits=logits,
                times=nll_label.reshape(-1, 1),
                censorships=censorship.reshape(-1, 1),
            )["loss"]
            weighted = aux_nll * float(weight)
            auxiliary_total = weighted if auxiliary_total is None else auxiliary_total + weighted
            log_dict[f"aux_{term_name}_nll"] = float(aux_nll.detach().item())
            log_dict[f"weighted_aux_{term_name}_nll"] = float(weighted.detach().item())

        if auxiliary_total is not None:
            results["auxiliary_loss"] = auxiliary_total
            if results.get("loss") is None:
                results["loss"] = auxiliary_total
            else:
                results["loss"] = results["loss"] + auxiliary_total
            log_dict["loss"] = float(results["loss"].detach().item())

    def forward(
        self,
        data,
        omics=None,
        *,
        attn_mask=None,
        label=None,
        discrete_label=None,
        censorship=None,
        survival_time=None,
        loss_fn=None,
        return_attn=False,
        batch=None,
        defer_survival_loss=False,
        include_auxiliary_loss=True,
        **_: Any,
    ):
        if not self.supports_batched_patients:
            data = _remove_single_patient_padding(data, attn_mask, self.model_name)
        logits, payload = self._call_backbone(
            data, omics, return_attn=return_attn, batch=batch
        )
        results, log_dict = process_surv(
            logits,
            label,
            censorship,
            loss_fn,
            survival_time=survival_time,
            defer_survival_loss=defer_survival_loss,
        )
        self._add_auxiliary_losses(
            results,
            log_dict,
            payload,
            loss_fn=loss_fn,
            label=label,
            discrete_label=discrete_label,
            aux_nll_loss=self.aux_nll_loss,
            censorship=censorship,
            enabled=(
                self.training
                and loss_fn is not None
                and include_auxiliary_loss
            ),
        )
        # Keep useful model outputs available without allowing a backbone's
        # precomputed risk/loss fields to overwrite the shared survival policy.
        for key, value in payload.items():
            if key not in {"logits", "risk", "loss"}:
                results[key] = value
        return results, log_dict


__all__ = ["UnifiedSurvivalAdapter"]
