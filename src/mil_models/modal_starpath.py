"""Joint coarse-to-fine STARPath survival backbone.

This single module owns the complete runtime model: coarse conditioning,
morphology-space regions, pathway encoders, morphology-anchored ST routing,
regional UOT, TITAN memory interaction, and patient classification.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import pickle
from collections import defaultdict
import math
import os
from os import PathLike
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F


EXPECTED_MORPHOLOGY_PROTOTYPES = 16
EXPECTED_PATCH_DIM = 768


def load_morphology_centroids(
    prototype_path: str,
    *,
    expected_count: int = EXPECTED_MORPHOLOGY_PROTOTYPES,
    expected_dim: int = EXPECTED_PATCH_DIM,
) -> torch.Tensor:
    """Load and validate one fold-specific MMP/PANTHER centroid artifact."""
    if prototype_path is None:
        raise ValueError("STARPath requires a fold-specific prototype_path")
    path = os.path.realpath(os.path.abspath(os.path.expanduser(os.fspath(prototype_path))))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"STARPath prototype file not found: {path}")

    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping) or "prototypes" not in payload:
        raise ValueError(
            "STARPath prototype pickle must contain a 'prototypes' array"
        )

    values = np.asarray(payload["prototypes"])
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    expected_shape = (int(expected_count), int(expected_dim))
    if values.shape != expected_shape:
        raise ValueError(
            f"STARPath prototypes must have shape {expected_shape}, got {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("STARPath prototypes must be floating point")
    if not np.isfinite(values).all():
        raise ValueError("STARPath prototypes contain NaN/Inf values")
    return torch.from_numpy(np.array(values, dtype=np.float32, copy=True))


def nearest_morphology_assignments(
    features: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Assign every raw CONCH feature to its nearest Euclidean centroid."""
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError("features must be a two-dimensional tensor")
    if not torch.is_tensor(centroids) or centroids.ndim != 2:
        raise ValueError("centroids must be a two-dimensional tensor")
    if features.shape[0] == 0:
        raise ValueError("Cannot assign an empty patch bag")
    if features.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"Feature/centroid width mismatch: {features.shape[1]} vs {centroids.shape[1]}"
        )
    if not bool(torch.isfinite(features).all().detach().item()):
        raise ValueError("Patch features contain NaN/Inf values")
    if not bool(torch.isfinite(centroids).all().detach().item()):
        raise ValueError("Morphology centroids contain NaN/Inf values")

    # Assignment is deliberately discrete and the vocabulary is frozen.  Use
    # float32 for stable distances even under mixed-precision model execution.
    with torch.no_grad():
        x = features.detach().to(dtype=torch.float32)
        mu = centroids.detach().to(device=x.device, dtype=torch.float32)
        squared_distance = (
            x.square().sum(dim=1, keepdim=True)
            + mu.square().sum(dim=1).unsqueeze(0)
            - 2.0 * (x @ mu.transpose(0, 1))
        )
        return squared_distance.argmin(dim=1)


def summarize_full_slide_morphology(
    features: torch.Tensor,
    centroids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return padded full-slide tokens, occupancies, and validity.

    The returned tensors have shapes ``[M, D]``, ``[M]``, and ``[M]``.  Empty
    prototypes are zero-filled and marked invalid; callers exclude them from
    KL-UOT while retaining the original prototype IDs for sampled-patch lookup.
    """
    assignments = nearest_morphology_assignments(features, centroids)
    prototype_count = int(centroids.shape[0])
    counts = torch.bincount(assignments, minlength=prototype_count)
    valid = counts > 0

    tokens = features.new_zeros((prototype_count, features.shape[1]))
    tokens.index_add_(0, assignments.to(device=features.device), features)
    denominator = counts.to(device=features.device, dtype=features.dtype).clamp_min(1)
    tokens = tokens / denominator.unsqueeze(1)
    tokens = torch.where(valid.to(features.device).unsqueeze(1), tokens, torch.zeros_like(tokens))

    occupancy = counts.to(device=features.device, dtype=features.dtype)
    occupancy = occupancy / float(features.shape[0])
    return tokens, occupancy, valid.to(device=features.device)


def summarize_full_slide_morphology_route_atlas(
    features: torch.Tensor,
    st: torch.Tensor,
    centroids: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Summarize full-slide morphology and aligned pseudo-ST in one pass.

    Every patch is assigned to the same frozen morphology vocabulary used by
    coarse STARPath. For each morphology anchor, pseudo-ST is averaged over
    *all* aligned slide patches, before any TITAN bag sampling takes place.
    The six returned tensors have shapes ``[M,D]``, ``[M]``, ``[M]``,
    ``[M,G_st]``, ``[M]``, and ``[M]`` respectively. Empty anchor slots are
    represented by exact zeros and a false validity bit.

    Keeping this operation beside :func:`summarize_full_slide_morphology`
    guarantees that the coarse summary and route atlas use one identical hard
    assignment rather than two independently reimplemented definitions.
    """
    if not torch.is_tensor(st) or st.ndim != 2:
        raise ValueError("st must be a two-dimensional tensor")
    if not st.is_floating_point():
        raise TypeError("st must be floating point")
    if st.shape[0] != features.shape[0]:
        raise ValueError(
            "Full-slide WSI/ST patch counts differ: "
            f"{features.shape[0]} vs {st.shape[0]}"
        )
    if st.shape[1] == 0:
        raise ValueError("st must contain at least one gene")
    if not bool(torch.isfinite(st).all().detach().item()):
        raise ValueError("ST predictions contain NaN/Inf values")

    assignments = nearest_morphology_assignments(features, centroids)
    prototype_count = int(centroids.shape[0])
    counts = torch.bincount(assignments, minlength=prototype_count)
    valid = counts > 0

    morphology_tokens = features.new_zeros(
        (prototype_count, features.shape[1])
    )
    morphology_tokens.index_add_(
        0, assignments.to(device=features.device), features
    )
    morphology_denominator = counts.to(
        device=features.device, dtype=features.dtype
    ).clamp_min(1)
    morphology_tokens = morphology_tokens / morphology_denominator.unsqueeze(1)
    morphology_valid = valid.to(device=features.device)
    morphology_tokens = torch.where(
        morphology_valid.unsqueeze(1),
        morphology_tokens,
        torch.zeros_like(morphology_tokens),
    )
    morphology_occupancy = counts.to(
        device=features.device, dtype=features.dtype
    ) / float(features.shape[0])

    route_assignments = assignments.to(device=st.device)
    route_counts = counts.to(device=st.device)
    route_valid = route_counts > 0
    route_atlas = st.new_zeros((prototype_count, st.shape[1]))
    route_atlas.index_add_(0, route_assignments, st)
    route_denominator = route_counts.to(dtype=st.dtype).clamp_min(1)
    route_atlas = route_atlas / route_denominator.unsqueeze(1)
    route_atlas = torch.where(
        route_valid.unsqueeze(1), route_atlas, torch.zeros_like(route_atlas)
    )

    return (
        morphology_tokens,
        morphology_occupancy,
        morphology_valid,
        route_atlas,
        route_counts,
        route_valid,
    )



DEFAULT_TITAN_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "TITAN_STARPath"
)

def _load_titan(model_path: str) -> nn.Module:
    """Load the private encoder only when a STARPath model is constructed."""
    from .TITAN_STARPath.modeling_titan import Titan

    return Titan.from_pretrained(model_path, local_files_only=True)


# Shared validation and numerical helpers.
def absolute_path(value: os.PathLike | str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(os.fspath(value))))


def parse_patch_size_lv0(value: Any) -> int:
    """Parse one positive integer-valued level-0 patch size."""
    if value is None:
        raise ValueError("STARPath patch_size_lv0 is required")
    scalar = value
    if torch.is_tensor(scalar):
        if scalar.numel() != 1:
            raise ValueError("STARPath patch_size_lv0 must contain one value")
        scalar = scalar.detach().cpu().reshape(-1)[0].item()
    elif isinstance(scalar, np.ndarray):
        if scalar.size != 1:
            raise ValueError("STARPath patch_size_lv0 must contain one value")
        scalar = scalar.reshape(-1)[0].item()
    elif isinstance(scalar, np.generic):
        scalar = scalar.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise TypeError("STARPath patch_size_lv0 must be an integer, not bool")
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("STARPath patch_size_lv0 must be numeric") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(
            "STARPath patch_size_lv0 must be finite and integer-valued, "
            f"got {scalar}"
        )
    parsed = int(numeric)
    if parsed <= 0:
        raise ValueError("STARPath patch_size_lv0 must be positive")
    return parsed


def normalise_gene_axis(values: Optional[Sequence[str]], modality: str) -> tuple[str, ...]:
    if values is None:
        raise ValueError(f"STARPath requires {modality}_gene_seq")
    axis = tuple(str(value).strip() for value in values)
    if not axis or any(not value for value in axis):
        raise ValueError(f"STARPath received an empty or invalid {modality} gene axis")
    return axis


def validate_runtime_gene_axis(
    values: Optional[Sequence[str]],
    expected: Sequence[str],
    modality: str,
) -> tuple[str, ...]:
    if values is None:
        return tuple(expected)
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (list, tuple)) and len(values) == 1:
        first = values[0]
        if isinstance(first, np.ndarray):
            first = first.tolist()
        if isinstance(first, (list, tuple)):
            values = first
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise TypeError(f"{modality}_gene_seq must be a sequence of gene names")
    if any(isinstance(value, (list, tuple, np.ndarray)) for value in values):
        raise ValueError(f"STARPath expects one {modality}_gene_seq per forward call")
    runtime = tuple(str(gene).strip() for gene in values)
    if runtime != tuple(expected):
        raise ValueError(
            f"Batch {modality.upper()} gene sequence differs from the "
            "framework-initialized sequence"
        )
    return runtime


def validate_forward_inputs(
    *,
    x1: torch.Tensor,
    coords: torch.Tensor,
    st: torch.Tensor,
    rna: torch.Tensor,
    expected_st_gene_seq: Sequence[str],
    expected_rna_gene_seq: Sequence[str],
    st_gene_seq: Optional[Sequence[str]],
    rna_gene_seq: Optional[Sequence[str]],
    input_dim: int,
    patch_size_lv0: Any,
    model_device: Optional[torch.device] = None,
    model_dtype: Optional[torch.dtype] = None,
):
    if not torch.is_tensor(x1):
        raise TypeError("STARPath x1 must be a torch.Tensor")
    if x1.ndim != 2 or x1.shape[0] == 0 or x1.shape[1] != input_dim:
        raise ValueError(
            f"STARPath x1 must have non-empty shape [N,{input_dim}], "
            f"got {tuple(x1.shape)}"
        )
    if not x1.is_floating_point():
        raise TypeError("STARPath x1 must be floating point")
    if not bool(torch.isfinite(x1).all().detach().item()):
        raise ValueError("STARPath x1 contains NaN/Inf values")
    if not torch.is_tensor(coords):
        raise TypeError("STARPath coords must be a torch.Tensor")
    if coords.ndim != 2 or tuple(coords.shape) != (x1.shape[0], 2):
        raise ValueError(
            f"STARPath coords must have shape [{x1.shape[0]},2], "
            f"got {tuple(coords.shape)}"
        )
    if not bool(torch.isfinite(coords).all().detach().item()):
        raise ValueError("STARPath coords contains NaN/Inf values")
    if coords.dtype == torch.bool:
        raise TypeError("STARPath coords must contain integer-valued coordinates")
    coords_float64 = coords.detach().to(dtype=torch.float64)
    if not bool((coords_float64 == coords_float64.round()).all().item()):
        raise ValueError(
            "STARPath coords must be integer-valued level-0 pixel coordinates"
        )
    int64_info = torch.iinfo(torch.int64)
    if bool(
        (
            (coords_float64 < int64_info.min)
            | (coords_float64 > int64_info.max)
        ).any().item()
    ):
        raise ValueError("STARPath coords exceed the int64 coordinate range")
    if st is None or not torch.is_tensor(st):
        raise TypeError("STARPath requires ST as a torch.Tensor")
    if st.ndim != 2 or st.shape[0] != x1.shape[0]:
        raise ValueError(
            f"STARPath st must have shape [N,G_st] aligned with x1, got {tuple(st.shape)}"
        )
    if st.shape[1] != len(expected_st_gene_seq):
        raise ValueError(
            f"STARPath st width {st.shape[1]} differs from the configured gene "
            f"axis ({len(expected_st_gene_seq)})"
        )
    if not bool(torch.isfinite(st).all().detach().item()):
        raise ValueError("STARPath st contains NaN/Inf values")
    if rna is None or not torch.is_tensor(rna):
        raise TypeError("STARPath requires bulk RNA as a torch.Tensor")
    if rna.ndim == 2 and rna.shape[0] == 1:
        rna = rna.squeeze(0)
    if rna.ndim != 1 or rna.shape[0] != len(expected_rna_gene_seq):
        raise ValueError(
            "STARPath rna must have shape [G_rna] or [1,G_rna] matching the "
            f"configured axis; got {tuple(rna.shape)}"
        )
    if not bool(torch.isfinite(rna).all().detach().item()):
        raise ValueError("STARPath rna contains NaN/Inf values")
    validate_runtime_gene_axis(st_gene_seq, expected_st_gene_seq, "st")
    validate_runtime_gene_axis(rna_gene_seq, expected_rna_gene_seq, "rna")
    target_device = x1.device if model_device is None else model_device
    target_dtype = x1.dtype if model_dtype is None else model_dtype
    if not target_dtype.is_floating_point:
        raise TypeError("STARPath model dtype must be floating point")
    return (
        x1.to(device=target_device, dtype=target_dtype),
        coords.to(device=target_device, dtype=torch.int64),
        st.to(device=target_device, dtype=target_dtype),
        rna.to(device=target_device, dtype=target_dtype),
        parse_patch_size_lv0(patch_size_lv0),
    )


def parse_injection_layers(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("titan_inject_layers must be an int, sequence, or delimited string")
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    elif isinstance(value, (int, np.integer)):
        values = [value]
    elif isinstance(value, Iterable):
        values = list(value)
    else:
        raise TypeError("titan_inject_layers must be an int, sequence, or delimited string")
    if any(isinstance(layer, (bool, np.bool_)) for layer in values):
        raise TypeError("titan_inject_layers must contain integers, not bool")
    try:
        numeric_layers = [float(layer) for layer in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("titan_inject_layers must contain only integers") from exc
    if any(not math.isfinite(layer) or not layer.is_integer() for layer in numeric_layers):
        raise ValueError("titan_inject_layers must contain only finite integers")
    return sorted({int(layer) for layer in numeric_layers})



def parse_trainable_layers(value: Any) -> list[int]:
    """Parse an independent TITAN trainable-layer specification."""
    if value is None:
        return []
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return []
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("titan_trainable_layers must contain integers, not bool")
    try:
        return parse_injection_layers(value)
    except (TypeError, ValueError) as exc:
        message = str(exc).replace("titan_inject_layers", "titan_trainable_layers")
        raise type(exc)(message) from exc

def bounded_logit(initial: float, maximum: float) -> torch.Tensor:
    if maximum <= 0:
        raise ValueError("A bounded residual maximum must be positive")
    ratio = min(max(float(initial) / float(maximum), 1e-6), 1.0 - 1e-6)
    return torch.tensor(float(math.log(ratio / (1.0 - ratio))))


def init_new_module(module: nn.Module) -> None:
    """Initialize a new adapter without touching pretrained TITAN weights."""
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_normal_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.LayerNorm):
            if child.weight is not None:
                nn.init.ones_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def nonnegative_weight(value: Any, name: str) -> float:
    weight = float(value)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return weight

# Coarse pathway encoding and transport primitives.
EXPECTED_HALLMARK_PATHWAYS = 50
def _snn_block(dim_in: int, dim_out: int, dropout: float) -> nn.Sequential:
    """The Linear-ELU-AlphaDropout block used by MMP pathway encoders."""
    return nn.Sequential(
        nn.Linear(int(dim_in), int(dim_out)),
        nn.ELU(),
        nn.AlphaDropout(p=float(dropout), inplace=False),
    )


class HallmarkPathwayEncoder(nn.Module):
    """Fifty independent MMP-style SNNs over variable-width gene vectors."""

    def __init__(
        self,
        omic_sizes: Sequence[int],
        *,
        output_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        sizes = tuple(int(size) for size in omic_sizes)
        if len(sizes) != EXPECTED_HALLMARK_PATHWAYS:
            raise ValueError(
                "Coarse STARPath requires exactly 50 Hallmark pathways; "
                f"received {len(sizes)}"
            )
        if any(size <= 0 for size in sizes):
            raise ValueError("Every Hallmark pathway must contain at least one gene")
        if int(output_dim) <= 0:
            raise ValueError("pathway output_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("pathway dropout must lie in [0, 1)")

        self.omic_sizes = sizes
        self.output_dim = int(output_dim)
        self.networks = nn.ModuleList(
            nn.Sequential(
                _snn_block(size, self.output_dim, dropout),
                _snn_block(self.output_dim, self.output_dim, dropout),
            )
            for size in sizes
        )

    def forward(self, pathways: Sequence[torch.Tensor]) -> torch.Tensor:
        if not isinstance(pathways, (list, tuple)):
            raise TypeError("STARPath omics must be a sequence of 50 pathway tensors")
        if len(pathways) != len(self.networks):
            raise ValueError(
                f"Expected {len(self.networks)} pathway tensors, got {len(pathways)}"
            )

        encoded = []
        for index, (values, expected_width, network) in enumerate(
            zip(pathways, self.omic_sizes, self.networks)
        ):
            if not torch.is_tensor(values):
                raise TypeError(f"Pathway {index} is not a torch.Tensor")
            if values.ndim == 1:
                values = values.unsqueeze(0)
            elif values.ndim >= 2:
                values = values.reshape(-1, values.shape[-1])
            if values.ndim != 2 or values.shape[0] != 1:
                raise ValueError(
                    "Coarse STARPath encodes one patient at a time; pathway "
                    f"{index} has shape {tuple(values.shape)}"
                )
            if values.shape[1] != expected_width:
                raise ValueError(
                    f"Pathway {index} width {values.shape[1]} != {expected_width}"
                )
            if not values.is_floating_point():
                values = values.float()
            if not bool(torch.isfinite(values).all().detach().item()):
                raise ValueError(f"Pathway {index} contains NaN/Inf values")
            encoded.append(network(values).squeeze(0))
        return torch.stack(encoded, dim=0)


def plain_entropy_kl_uot(
    cost: torch.Tensor,
    source_reference: torch.Tensor,
    target_reference: torch.Tensor,
    *,
    epsilon: float = 0.07,
    tau_source: float = 0.5,
    tau_target: float = 0.5,
    iterations: int = 50,
) -> torch.Tensor:
    """Solve the exact plain-entropy KL-UOT objective in log space.

    The entropy kernel is strictly ``exp(-cost / epsilon)``.  In particular,
    the source/target reference product is *not* folded into the kernel.
    """
    if not torch.is_tensor(cost) or cost.ndim != 2 or cost.numel() == 0:
        raise ValueError("KL-UOT cost must be a non-empty matrix")
    if not cost.is_floating_point():
        raise TypeError("KL-UOT cost must be floating point")
    if source_reference.shape != (cost.shape[0],):
        raise ValueError("KL-UOT source reference shape does not match cost")
    if target_reference.shape != (cost.shape[1],):
        raise ValueError("KL-UOT target reference shape does not match cost")
    if float(epsilon) <= 0 or float(tau_source) <= 0 or float(tau_target) <= 0:
        raise ValueError("KL-UOT epsilon and marginal penalties must be positive")
    if int(iterations) <= 0:
        raise ValueError("KL-UOT iterations must be positive")
    if not bool(torch.isfinite(cost).all().detach().item()):
        raise ValueError("KL-UOT cost contains NaN/Inf values")

    work_dtype = (
        torch.float32
        if cost.dtype in {torch.float16, torch.bfloat16}
        else cost.dtype
    )
    work_cost = cost.to(dtype=work_dtype)
    a = source_reference.to(device=cost.device, dtype=work_dtype)
    b = target_reference.to(device=cost.device, dtype=work_dtype)
    if not bool((torch.isfinite(a) & (a > 0)).all().detach().item()):
        raise ValueError("KL-UOT source reference must be finite and strictly positive")
    if not bool((torch.isfinite(b) & (b > 0)).all().detach().item()):
        raise ValueError("KL-UOT target reference must be finite and strictly positive")

    eps = float(epsilon)
    source_power = float(tau_source) / (float(tau_source) + eps)
    target_power = float(tau_target) / (float(tau_target) + eps)
    log_kernel = -work_cost / eps
    log_a = torch.log(a)
    log_b = torch.log(b)
    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)

    for _ in range(int(iterations)):
        log_u = source_power * (
            log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        )
        log_v = target_power * (
            log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
        )

    log_transport = log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)
    transport = torch.exp(log_transport)
    if not bool(torch.isfinite(transport).all().detach().item()):
        raise FloatingPointError("KL-UOT produced NaN/Inf transport mass")
    # Keep the solver precision. Casting a valid plain-entropy solution back
    # to FP16/BF16 can round its small, strictly positive entries to zero. The
    # context is converted to the patch dtype only after the FP32 T^T V sum.
    return transport


def transported_molecular_context(
    transport: torch.Tensor,
    molecular_values: torch.Tensor,
) -> torch.Tensor:
    """Compute the deliberately unnormalized prototype molecular context."""
    if transport.ndim != 2 or molecular_values.ndim != 2:
        raise ValueError("transport and molecular_values must be matrices")
    if transport.shape[0] != molecular_values.shape[0]:
        raise ValueError("transport/value pathway dimensions differ")
    values = molecular_values.to(
        device=transport.device,
        dtype=transport.dtype,
    )
    return transport.transpose(0, 1) @ values

# TITAN-free coarse Path2Space conditioner.
class CoarseConditioner(nn.Module):
    """Condition raw patch features without owning TITAN or a classifier.

    Bulk-RNA Hallmark tokens are matched to a fixed, full-slide morphology
    summary with plain-entropy KL-UOT.  The resulting prototype contexts are
    looked up for each sampled raw patch and added through a bounded residual.
    """

    def __init__(
        self,
        *,
        omic_sizes: Sequence[int],
        prototype_path: Optional[str] = None,
        pathway_names: Optional[Sequence[str]] = None,
        input_dim: int = EXPECTED_PATCH_DIM,
        pathway_dim: int = 256,
        compatibility_dim: int = 256,
        context_dim: int = 256,
        pathway_dropout: float = 0.25,
        uot_epsilon: float = 0.07,
        uot_tau_source: float = 0.5,
        uot_tau_target: float = 0.5,
        uot_iterations: int = 50,
        alpha_init: float = 0.03,
        alpha_max: float = 0.10,
        morphology_centroids: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if int(input_dim) != EXPECTED_PATCH_DIM:
            raise ValueError(
                "CoarseConditioner requires 768-dimensional CONCH features"
            )
        if int(compatibility_dim) <= 0 or int(context_dim) <= 0:
            raise ValueError("coarse projection dimensions must be positive")
        if not 0.0 < float(alpha_max) <= 0.1:
            raise ValueError("coarse alpha_max must lie in (0,0.1]")
        if not 0.0 < float(alpha_init) < float(alpha_max):
            raise ValueError(
                "coarse alpha_init must lie strictly between zero and alpha_max"
            )

        self.input_dim = int(input_dim)
        self.pathway_dim = int(pathway_dim)
        self.context_dim = int(context_dim)
        self.uot_epsilon = float(uot_epsilon)
        self.uot_tau_source = float(uot_tau_source)
        self.uot_tau_target = float(uot_tau_target)
        self.uot_iterations = int(uot_iterations)
        self.alpha_max = float(alpha_max)

        if pathway_names is None:
            pathway_names = tuple(
                f"HALLMARK_{index:02d}"
                for index in range(EXPECTED_HALLMARK_PATHWAYS)
            )
        self.pathway_names = tuple(str(name) for name in pathway_names)
        if len(self.pathway_names) != EXPECTED_HALLMARK_PATHWAYS:
            raise ValueError(
                "coarse pathway_names must contain exactly 50 entries"
            )
        if len(set(self.pathway_names)) != len(self.pathway_names):
            raise ValueError("coarse pathway_names must be unique")

        self.pathway_encoder = HallmarkPathwayEncoder(
            omic_sizes,
            output_dim=self.pathway_dim,
            dropout=float(pathway_dropout),
        )
        self.rna_compatibility = nn.Linear(
            self.pathway_dim, int(compatibility_dim)
        )
        self.morphology_compatibility = nn.Linear(
            self.input_dim, int(compatibility_dim)
        )
        self.rna_value = nn.Linear(self.pathway_dim, self.context_dim)
        self.context_to_patch = nn.Linear(
            self.context_dim, self.input_dim, bias=False
        )
        self.gate = nn.Linear(
            self.input_dim + self.context_dim, self.input_dim
        )
        self.raw_alpha = nn.Parameter(
            bounded_logit(float(alpha_init), self.alpha_max)
        )

        if morphology_centroids is None:
            morphology_centroids = load_morphology_centroids(prototype_path)
        else:
            morphology_centroids = torch.as_tensor(
                morphology_centroids
            ).detach().float()
        expected_shape = (
            EXPECTED_MORPHOLOGY_PROTOTYPES,
            EXPECTED_PATCH_DIM,
        )
        if tuple(morphology_centroids.shape) != expected_shape:
            raise ValueError(
                "morphology_centroids must have shape "
                f"{expected_shape}, got {tuple(morphology_centroids.shape)}"
            )
        if not bool(torch.isfinite(morphology_centroids).all().item()):
            raise ValueError("morphology_centroids contain NaN/Inf values")
        self.register_buffer(
            "morphology_centroids",
            morphology_centroids.clone(),
            persistent=True,
        )

        for module in (
            self.pathway_encoder,
            self.rna_compatibility,
            self.morphology_compatibility,
            self.rna_value,
            self.context_to_patch,
            self.gate,
        ):
            init_new_module(module)

    @property
    def coarse_alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.raw_alpha)

    def encode_rna_pathways(
        self, pathways: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Encode the coarse RNA view once per patient."""
        return self.pathway_encoder(pathways)

    def _validate_inputs(
        self,
        patch_features: torch.Tensor,
        pathway_tokens: torch.Tensor,
        morphology_tokens: torch.Tensor,
        morphology_occupancy: torch.Tensor,
        morphology_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            not torch.is_tensor(patch_features)
            or patch_features.ndim != 2
            or patch_features.shape[0] == 0
            or patch_features.shape[1] != self.input_dim
        ):
            raise ValueError(
                f"patch_features must be non-empty [N,{self.input_dim}]"
            )
        if not patch_features.is_floating_point():
            raise TypeError("patch_features must be floating point")
        if not torch.is_tensor(pathway_tokens):
            raise TypeError("coarse_pathway_tokens must be a torch.Tensor")
        if pathway_tokens.ndim == 3 and pathway_tokens.shape[0] == 1:
            pathway_tokens = pathway_tokens.squeeze(0)
        expected_pathway_shape = (
            EXPECTED_HALLMARK_PATHWAYS,
            self.pathway_dim,
        )
        if tuple(pathway_tokens.shape) != expected_pathway_shape:
            raise ValueError(
                "coarse_pathway_tokens must have shape "
                f"{expected_pathway_shape}, got {tuple(pathway_tokens.shape)}"
            )
        expected_morphology_shape = (
            EXPECTED_MORPHOLOGY_PROTOTYPES,
            self.input_dim,
        )
        if (
            not torch.is_tensor(morphology_tokens)
            or tuple(morphology_tokens.shape) != expected_morphology_shape
        ):
            raise ValueError(
                "morphology_tokens must have shape "
                f"{expected_morphology_shape}"
            )
        if (
            not torch.is_tensor(morphology_occupancy)
            or tuple(morphology_occupancy.shape)
            != (EXPECTED_MORPHOLOGY_PROTOTYPES,)
        ):
            raise ValueError("morphology_occupancy must contain 16 entries")
        if (
            not torch.is_tensor(morphology_valid)
            or tuple(morphology_valid.shape)
            != (EXPECTED_MORPHOLOGY_PROTOTYPES,)
        ):
            raise ValueError("morphology_valid must contain 16 entries")

        device, dtype = patch_features.device, patch_features.dtype
        pathway_tokens = pathway_tokens.to(device=device, dtype=dtype)
        morphology_tokens = morphology_tokens.to(device=device, dtype=dtype)
        occupancy = morphology_occupancy.to(device=device, dtype=dtype)
        valid = morphology_valid.to(device=device, dtype=torch.bool)
        for name, values in (
            ("patch_features", patch_features),
            ("coarse_pathway_tokens", pathway_tokens),
            ("morphology_tokens", morphology_tokens),
            ("morphology_occupancy", occupancy),
        ):
            if not bool(torch.isfinite(values).all().detach().item()):
                raise ValueError(f"{name} contains NaN/Inf values")
        if not bool(valid.any().detach().item()):
            raise ValueError(
                "Full-slide morphology summary has no occupied prototype"
            )
        if bool((occupancy[valid] <= 0).any().detach().item()):
            raise ValueError(
                "Occupied morphology prototypes require positive mass"
            )
        if bool((occupancy[~valid] != 0).any().detach().item()):
            raise ValueError(
                "Absent morphology prototypes must have zero occupancy"
            )
        if not torch.allclose(
            occupancy.sum().detach(),
            occupancy.new_tensor(1.0),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("Full-slide morphology occupancy must sum to one")
        return pathway_tokens, morphology_tokens, occupancy, valid

    def forward(
        self,
        patch_features: torch.Tensor,
        *,
        pathway_tokens: torch.Tensor,
        morphology_tokens: torch.Tensor,
        morphology_occupancy: torch.Tensor,
        morphology_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        (
            pathway_tokens,
            morphology_tokens,
            occupancy,
            valid,
        ) = self._validate_inputs(
            patch_features,
            pathway_tokens,
            morphology_tokens,
            morphology_occupancy,
            morphology_valid,
        )
        device, dtype = patch_features.device, patch_features.dtype
        active_ids = torch.nonzero(valid, as_tuple=False).squeeze(1)
        active_morphology = morphology_tokens.index_select(0, active_ids)
        target_mass = occupancy.index_select(0, active_ids)
        source_mass = patch_features.new_full(
            (EXPECTED_HALLMARK_PATHWAYS,),
            1.0 / EXPECTED_HALLMARK_PATHWAYS,
        )

        rna_key = F.normalize(
            self.rna_compatibility(pathway_tokens), dim=-1
        )
        morphology_key = F.normalize(
            self.morphology_compatibility(active_morphology), dim=-1
        )
        cost = 1.0 - rna_key @ morphology_key.transpose(0, 1)
        transport = plain_entropy_kl_uot(
            cost,
            source_mass,
            target_mass,
            epsilon=self.uot_epsilon,
            tau_source=self.uot_tau_source,
            tau_target=self.uot_tau_target,
            iterations=self.uot_iterations,
        )
        active_context = transported_molecular_context(
            transport, self.rna_value(pathway_tokens)
        ).to(dtype=dtype)
        all_context = patch_features.new_zeros(
            (EXPECTED_MORPHOLOGY_PROTOTYPES, self.context_dim)
        ).index_copy(0, active_ids, active_context)

        sampled_assignments = nearest_morphology_assignments(
            patch_features, self.morphology_centroids
        ).to(device=device)
        patch_context = all_context.index_select(0, sampled_assignments)
        gate = torch.sigmoid(
            self.gate(torch.cat([patch_features, patch_context], dim=-1))
        )
        residual = (
            self.coarse_alpha.to(dtype=dtype)
            * gate
            * self.context_to_patch(patch_context)
        )
        conditioned = patch_features + residual

        input_rms = patch_features.square().mean().sqrt().clamp_min(1e-8)
        diagnostics: Dict[str, Any] = {
            "pathway_names": self.pathway_names,
            "active_morphology_ids": active_ids.detach(),
            "morphology_occupancy": occupancy.detach(),
            "sampled_morphology_ids": sampled_assignments.detach(),
            "transport": transport.detach(),
            "transport_cost": cost.detach(),
            "transport_row_mass": transport.sum(dim=1).detach(),
            "transport_col_mass": transport.sum(dim=0).detach(),
            "transport_total_mass": transport.sum().detach(),
            "prototype_molecular_context": all_context.detach(),
            "coarse_alpha": self.coarse_alpha.detach(),
            "residual_rms_ratio": (
                residual.square().mean().sqrt() / input_rms
            ).detach(),
        }
        return conditioned, diagnostics

    def condition_patch_features(
        self,
        *,
        patch_features: torch.Tensor,
        pathway_tokens: torch.Tensor,
        morphology_tokens: torch.Tensor,
        morphology_occupancy: torch.Tensor,
        morphology_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Named public API for composing the conditioner with one backbone."""
        return self.forward(
            patch_features,
            pathway_tokens=pathway_tokens,
            morphology_tokens=morphology_tokens,
            morphology_occupancy=morphology_occupancy,
            morphology_valid=morphology_valid,
        )


# Fixed-slot RNA/ST pathway encoders.
@dataclass(frozen=True)
class RNAPathwayEncoding:
    tokens: torch.Tensor  # [K, d]
    valid: torch.Tensor  # bool [K]


@dataclass(frozen=True)
class RegionalSTPathwayEncoding:
    tokens: torch.Tensor  # [R, K, d]
    valid: torch.Tensor  # bool [R, K]
    reliability: torch.Tensor  # [K]


def _load_signatures(
    path: str | PathLike[str], expected_count: int | None
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    frame = pd.read_csv(path)
    if frame.shape[1] == 0:
        raise ValueError("SurvPath signature CSV has no pathway columns")
    if expected_count is not None and frame.shape[1] != int(expected_count):
        raise ValueError(
            f"Expected {int(expected_count)} signature columns, got {frame.shape[1]}"
        )
    names = tuple(str(column).strip() for column in frame.columns)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Signature pathway names must be non-empty and unique")

    signatures = []
    for column in frame.columns:
        genes, seen = [], set()
        for value in frame[column]:
            if pd.isna(value):
                continue
            gene = str(value).strip()
            if gene and gene not in seen:
                genes.append(gene)
                seen.add(gene)
        if not genes:
            raise ValueError(f"Signature {column!r} contains no genes")
        signatures.append(tuple(genes))
    return names, tuple(signatures)


class PathwaySNN(nn.Module):
    """Two independent SurvPath-style Linear/ELU/AlphaDropout blocks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        hidden_dim = int(output_dim if hidden_dim is None else hidden_dim)
        if min(int(input_dim), int(output_dim), hidden_dim) <= 0:
            raise ValueError("PathwaySNN dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.ELU(),
            nn.AlphaDropout(float(dropout)),
            nn.Linear(hidden_dim, int(output_dim)),
            nn.ELU(),
            nn.AlphaDropout(float(dropout)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class PathwaySNNBank(nn.Module):
    """One SNN per slot; invalid slots are retained but produce exact zeros."""

    def __init__(
        self,
        source_indices: Sequence[Sequence[int]],
        slot_valid: Sequence[bool],
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if not source_indices or len(source_indices) != len(slot_valid):
            raise ValueError("source_indices and slot_valid must have equal length")
        self.output_dim = int(output_dim)
        self.networks = nn.ModuleList()
        self._index_names: list[str] = []
        for slot, indices in enumerate(source_indices):
            safe = tuple(int(index) for index in indices)
            if not safe or any(index < 0 for index in safe):
                raise ValueError("Every slot requires non-negative safe source indices")
            name = f"_source_index_{slot:03d}"
            self.register_buffer(
                name, torch.tensor(safe, dtype=torch.long), persistent=False
            )
            self._index_names.append(name)
            self.networks.append(
                PathwaySNN(len(safe), output_dim, hidden_dim, dropout)
            )
        self.register_buffer(
            "slot_valid",
            torch.tensor(tuple(bool(value) for value in slot_valid), dtype=torch.bool),
            persistent=False,
        )
        init_new_module(self)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        leading = tuple(values.shape[:-1])
        outputs = []
        for slot, network in enumerate(self.networks):
            if not bool(self.slot_valid[slot].item()):
                outputs.append(values.new_zeros(*leading, self.output_dim))
                continue
            indices = getattr(self, self._index_names[slot])
            outputs.append(network(values.index_select(-1, indices)))
        return torch.stack(outputs, dim=-2)


class SurvPathPathwayEncoder(nn.Module):
    """Preserve CSV column order and independently map RNA and ST gene axes.

    All 50 Hallmark columns remain fixed slots.  If a signature has no genes
    on one modality's axis, index zero is stored as a safe gather placeholder,
    while its token is forced to zero and its validity is false.
    """

    def __init__(
        self,
        signature_path: str | PathLike[str],
        rna_gene_seq: Sequence[str],
        st_gene_seq: Sequence[str],
        token_dim: int = 256,
        hidden_dim: int | None = None,
        dropout: float = 0.25,
        expected_num_pathways: int | None = 50,
        min_rna_genes: int | None = None,
        min_st_genes: int | None = None,
        pathway_dim: int | None = None,
        min_genes: int = 1,
        min_st_reliability: float = 0.0,
    ) -> None:
        super().__init__()
        self.rna_gene_seq = normalise_gene_axis(rna_gene_seq, "rna")
        self.st_gene_seq = normalise_gene_axis(st_gene_seq, "st")
        if len(set(self.rna_gene_seq)) != len(self.rna_gene_seq):
            raise ValueError("RNA gene axis contains duplicates")
        if len(set(self.st_gene_seq)) != len(self.st_gene_seq):
            raise ValueError("ST gene axis contains duplicates")
        self.pathway_names, self.signature_genes = _load_signatures(
            signature_path, expected_num_pathways
        )
        if pathway_dim is not None:
            if int(token_dim) != 256 and int(token_dim) != int(pathway_dim):
                raise ValueError("token_dim and pathway_dim disagree")
            token_dim = int(pathway_dim)
        self.token_dim = int(token_dim)
        self.min_rna_genes = int(
            min_genes if min_rna_genes is None else min_rna_genes
        )
        self.min_st_genes = int(
            min_genes if min_st_genes is None else min_st_genes
        )
        self.min_st_reliability = float(min_st_reliability)
        if min(self.token_dim, self.min_rna_genes, self.min_st_genes) <= 0:
            raise ValueError("token dimension and valid-gene thresholds must be positive")
        if not 0.0 <= self.min_st_reliability <= 1.0:
            raise ValueError("min_st_reliability must lie in [0,1]")

        rna_axis = {gene: index for index, gene in enumerate(self.rna_gene_seq)}
        st_axis = {gene: index for index, gene in enumerate(self.st_gene_seq)}
        rna_genes, st_genes, rna_indices, st_indices = [], [], [], []
        for signature in self.signature_genes:
            current_rna = tuple(gene for gene in signature if gene in rna_axis)
            current_st = tuple(gene for gene in signature if gene in st_axis)
            rna_genes.append(current_rna)
            st_genes.append(current_st)
            rna_indices.append(tuple(rna_axis[gene] for gene in current_rna))
            st_indices.append(tuple(st_axis[gene] for gene in current_st))

        self.rna_source_genes = tuple(rna_genes)
        self.st_source_genes = tuple(st_genes)
        self.rna_available_counts = tuple(map(len, rna_indices))
        self.st_available_counts = tuple(map(len, st_indices))
        self.rna_source_indices = tuple(value or (0,) for value in rna_indices)
        self.st_source_indices = tuple(value or (0,) for value in st_indices)
        rna_valid = tuple(
            count >= self.min_rna_genes for count in self.rna_available_counts
        )
        reliability = tuple(
            count / len(signature)
            for count, signature in zip(self.st_available_counts, self.signature_genes)
        )
        st_valid = tuple(
            count >= self.min_st_genes
            and score >= self.min_st_reliability
            for count, score in zip(self.st_available_counts, reliability)
        )
        self.register_buffer(
            "rna_slot_valid", torch.tensor(rna_valid, dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "st_slot_valid", torch.tensor(st_valid, dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "st_reliability", torch.tensor(reliability), persistent=True
        )
        self.rna_bank = PathwaySNNBank(
            self.rna_source_indices, rna_valid, token_dim, hidden_dim, dropout
        )
        self.st_bank = PathwaySNNBank(
            self.st_source_indices, st_valid, token_dim, hidden_dim, dropout
        )

    @property
    def num_pathways(self) -> int:
        return len(self.pathway_names)

    @staticmethod
    def _validate(values: torch.Tensor, width: int, name: str) -> None:
        if not torch.is_tensor(values) or not values.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        if values.shape[-1] != width:
            raise ValueError(f"{name} must have gene width {width}")
        if not bool(torch.isfinite(values).all().detach().item()):
            raise ValueError(f"{name} contains NaN/Inf values")

    def encode_rna(self, rna: torch.Tensor) -> RNAPathwayEncoding:
        if torch.is_tensor(rna) and rna.ndim == 2 and rna.shape[0] == 1:
            rna = rna.squeeze(0)
        if not torch.is_tensor(rna) or rna.ndim != 1:
            raise ValueError("rna must have shape [G] or [1,G]")
        self._validate(rna, len(self.rna_gene_seq), "rna")
        valid = self.rna_slot_valid.to(rna.device)
        tokens = self.rna_bank(rna).masked_fill(~valid.unsqueeze(-1), 0.0)
        return RNAPathwayEncoding(tokens=tokens, valid=valid)

    def encode_regional_st(
        self, regional_st: torch.Tensor, active_region_mask: torch.Tensor
    ) -> RegionalSTPathwayEncoding:
        if not torch.is_tensor(regional_st) or regional_st.ndim != 2:
            raise ValueError("regional_st must have shape [R,G]")
        self._validate(regional_st, len(self.st_gene_seq), "regional_st")
        if (
            not torch.is_tensor(active_region_mask)
            or active_region_mask.ndim != 1
            or active_region_mask.shape[0] != regional_st.shape[0]
        ):
            raise ValueError("active_region_mask must have shape [R]")
        active = active_region_mask.to(regional_st.device, dtype=torch.bool)
        valid = active[:, None] & self.st_slot_valid.to(regional_st.device)[None, :]
        tokens = self.st_bank(regional_st).masked_fill(~valid[..., None], 0.0)
        reliability = self.st_reliability.to(
            device=regional_st.device, dtype=regional_st.dtype
        )
        return RegionalSTPathwayEncoding(tokens, valid, reliability)


RNAPathwayOutput = RNAPathwayEncoding
RegionalSTPathwayOutput = RegionalSTPathwayEncoding

# Deterministic morphology-space region construction.
@dataclass
class RegionOutput:
    """Fixed-width region-construction outputs for one slide.

    ``R`` is always ``region_num``.  Slots beyond ``min(region_num, N)`` are
    zero padded and marked false by ``active_region_mask``.
    """

    coords_normalized: Tensor       # [N, 2]
    active_region_mask: Tensor      # [R] bool
    seed_indices: Tensor            # [R] long; -1 for inactive slots
    candidate_mask: Tensor          # [N, R] bool
    q_soft_initial: Tensor          # [N, R]
    q_soft: Tensor                  # [N, R]
    q_hard: Tensor                  # [N, R]
    q_st: Tensor                    # [N, R], hard forward / soft backward
    refined_prototypes: Tensor      # [R, D]
    refined_centers: Tensor         # [R, 2]
    refined_sigma2: Tensor          # [R]
    region_counts: Tensor           # [R]
    occupancy: Tensor               # [R], sums to one over active slots
    visual_region: Tensor           # [R, D]
    center_region: Tensor           # [R, 2]
    loss_region: Tensor             # scalar
    loss_morph: Tensor              # scalar
    loss_coord: Tensor              # scalar

    @property
    def q(self) -> Tensor:
        """Compatibility alias for the differentiable hard assignment."""

        return self.q_st

    @property
    def occupancy_fraction(self) -> Tensor:
        """Compatibility alias used by earlier STARPath implementations."""

        return self.occupancy

    @property
    def active_mask(self) -> Tensor:
        return self.active_region_mask

    @property
    def count(self) -> Tensor:
        return self.region_counts

    @property
    def visual(self) -> Tensor:
        return self.visual_region

    @property
    def center(self) -> Tensor:
        return self.center_region


def _region_masked_softmax(logits: Tensor, mask: Tensor, dim: int, eps: float) -> Tensor:
    """Softmax that returns exact zeros for masked or fully masked rows."""

    mask = mask.to(device=logits.device, dtype=torch.bool)
    valid = mask.any(dim=dim, keepdim=True)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    masked_logits = torch.where(valid, masked_logits, torch.zeros_like(masked_logits))
    probs = F.softmax(masked_logits, dim=dim) * mask.to(dtype=logits.dtype)
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    return torch.where(valid, probs, torch.zeros_like(probs))


class MorphologySpaceRegionConstructor(nn.Module):
    """Build stable, fixed-width regions using morphology and coordinates only."""

    def __init__(
        self,
        input_dim: int = 768,
        region_num: int = 12,
        assignment_dim: int = 256,
        region_seed_knn: int = 8,
        region_candidate_topk: int = 3,
        region_temperature: float = 0.2,
        region_score_spatial_weight: float = 1.0,
        region_loss_spatial_weight: float = 1.0,
        region_min_spatial_scale: float = 0.05,
        region_norm_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.region_num = int(region_num)
        self.assignment_dim = int(assignment_dim)
        self.region_seed_knn = int(region_seed_knn)
        self.region_candidate_topk = int(region_candidate_topk)
        self.region_temperature = float(region_temperature)
        self.region_score_spatial_weight = float(region_score_spatial_weight)
        self.region_loss_spatial_weight = float(region_loss_spatial_weight)
        self.region_min_spatial_scale = float(region_min_spatial_scale)
        self.region_norm_epsilon = float(region_norm_epsilon)

        if self.input_dim <= 0 or self.assignment_dim <= 0:
            raise ValueError("input_dim and assignment_dim must be positive")
        if self.region_num <= 0:
            raise ValueError("region_num must be positive")
        if self.region_seed_knn <= 0 or self.region_candidate_topk <= 0:
            raise ValueError("region_seed_knn and region_candidate_topk must be positive")
        if self.region_temperature <= 0:
            raise ValueError("region_temperature must be positive")
        if self.region_min_spatial_scale <= 0 or self.region_norm_epsilon <= 0:
            raise ValueError("region spatial scale and epsilon must be positive")
        if self.region_score_spatial_weight < 0 or self.region_loss_spatial_weight < 0:
            raise ValueError("region spatial weights must be non-negative")

        self.patch_projection = nn.Linear(self.input_dim, self.assignment_dim)
        self.prototype_projection = nn.Linear(self.input_dim, self.assignment_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (self.patch_projection, self.prototype_projection):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def eps(self) -> float:
        return self.region_norm_epsilon

    def _validate_inputs(self, x_vis: Tensor, coords: Tensor) -> None:
        if not torch.is_tensor(x_vis) or not torch.is_tensor(coords):
            raise TypeError("x_vis and coords must be torch.Tensor objects")
        if x_vis.ndim != 2 or x_vis.shape[0] == 0 or x_vis.shape[1] != self.input_dim:
            raise ValueError(
                f"x_vis must have non-empty shape [N,{self.input_dim}], "
                f"got {tuple(x_vis.shape)}"
            )
        if not x_vis.is_floating_point():
            raise TypeError("x_vis must be floating point")
        if coords.ndim != 2 or tuple(coords.shape) != (x_vis.shape[0], 2):
            raise ValueError(
                f"coords must have shape [{x_vis.shape[0]},2], got {tuple(coords.shape)}"
            )
        if not bool(torch.isfinite(x_vis).all().detach().item()):
            raise ValueError("x_vis contains NaN/Inf values")
        if not bool(torch.isfinite(coords).all().detach().item()):
            raise ValueError("coords contains NaN/Inf values")

    def _normalize_coords(self, coords: Tensor, dtype: torch.dtype) -> Tensor:
        coords_f = coords.to(dtype=dtype)
        lower = coords_f.amin(dim=0, keepdim=True)
        upper = coords_f.amax(dim=0, keepdim=True)
        return (coords_f - lower) / (upper - lower + self.region_norm_epsilon)

    @staticmethod
    def _deterministic_fps(coords: Tensor, count: int) -> Tensor:
        """Unique coordinate FPS with deterministic first-index tie breaking."""

        patch_count = int(coords.shape[0])
        if not 1 <= int(count) <= patch_count:
            raise ValueError("FPS count must lie in [1, number of patches]")

        slide_center = coords.mean(dim=0, keepdim=True)
        first = torch.argmin(torch.cdist(coords, slide_center).squeeze(1))
        selected = torch.zeros(patch_count, dtype=torch.bool, device=coords.device)
        selected[first] = True
        indices = [first]
        min_distance = torch.cdist(coords, coords[first : first + 1]).squeeze(1)

        for _ in range(1, int(count)):
            eligible_distance = min_distance.masked_fill(selected, -torch.inf)
            next_index = torch.argmax(eligible_distance)
            selected[next_index] = True
            indices.append(next_index)
            next_distance = torch.cdist(
                coords, coords[next_index : next_index + 1]
            ).squeeze(1)
            min_distance = torch.minimum(min_distance, next_distance)

        return torch.stack(indices).long()

    def _candidate_mask(
        self,
        seed_distance2: Tensor,
        seed_indices: Tensor,
        region_count: int,
    ) -> Tensor:
        """Build a fixed seed-nearest mask and guarantee every seed-own pair."""

        patch_count = int(seed_distance2.shape[0])
        candidate_count = min(self.region_candidate_topk, int(region_count))
        candidate_indices = torch.argsort(
            seed_distance2, dim=1, stable=True
        )[:, :candidate_count]
        candidate_mask = torch.zeros(
            patch_count,
            int(region_count),
            dtype=torch.bool,
            device=seed_distance2.device,
        )
        candidate_mask.scatter_(1, candidate_indices, True)

        region_ids = torch.arange(region_count, device=seed_distance2.device)
        owns_seed = candidate_mask[seed_indices, region_ids]
        if bool((~owns_seed).any().detach().item()):
            rows = seed_indices[~owns_seed]
            regions = region_ids[~owns_seed]
            displaced = candidate_indices[rows, -1]
            candidate_mask[rows, displaced] = False
            candidate_mask[rows, regions] = True
        return candidate_mask

    def _assignment_score(
        self,
        patch_embedding: Tensor,
        prototypes: Tensor,
        coords: Tensor,
        centers: Tensor,
        sigma2: Tensor,
    ) -> Tensor:
        prototype_embedding = F.normalize(
            self.prototype_projection(prototypes),
            dim=-1,
            eps=self.region_norm_epsilon,
        )
        visual_score = patch_embedding @ prototype_embedding.transpose(0, 1)
        spatial_distance2 = torch.cdist(coords, centers).square()
        spatial_score = -spatial_distance2 / sigma2.clamp_min(
            self.region_min_spatial_scale ** 2
        ).unsqueeze(0)
        return visual_score + self.region_score_spatial_weight * spatial_score

    def forward(self, x_vis: Tensor, coords: Tensor) -> RegionOutput:
        self._validate_inputs(x_vis, coords)
        patch_count = int(x_vis.shape[0])
        effective_regions = min(self.region_num, patch_count)
        eps = self.region_norm_epsilon

        # Enforce the modality boundary even if a future caller supplies patch
        # features produced by a trainable upstream molecular branch.
        x_morph = x_vis.detach()
        coords_normalized = self._normalize_coords(coords, dtype=x_vis.dtype)
        seed_active = self._deterministic_fps(coords_normalized, effective_regions)
        seed_centers = coords_normalized.index_select(0, seed_active)
        seed_distance2 = torch.cdist(coords_normalized, seed_centers).square()

        seed_k = min(self.region_seed_knn, patch_count)
        seed_neighbours = torch.argsort(
            seed_distance2, dim=0, stable=True
        )[:seed_k]  # [k_seed, R_eff]
        initial_prototypes = x_morph[seed_neighbours].mean(dim=0)
        initial_sigma2 = seed_distance2.gather(0, seed_neighbours).mean(dim=0)
        initial_sigma2 = initial_sigma2.clamp_min(self.region_min_spatial_scale ** 2)

        candidate_active = self._candidate_mask(
            seed_distance2, seed_active, effective_regions
        )
        patch_embedding = F.normalize(
            self.patch_projection(x_morph), dim=-1, eps=eps
        )
        initial_score = self._assignment_score(
            patch_embedding,
            initial_prototypes,
            coords_normalized,
            seed_centers,
            initial_sigma2,
        )
        q_initial_active = _region_masked_softmax(
            initial_score / self.region_temperature,
            candidate_active,
            dim=1,
            eps=eps,
        )

        # Exactly one deterministic, stop-gradient refinement.  The candidate
        # graph remains fixed to the original coordinate-FPS seeds.
        initial_mass = q_initial_active.sum(dim=0)
        refined_prototypes_active = (
            q_initial_active.transpose(0, 1) @ x_morph
        ) / (initial_mass.unsqueeze(-1) + eps)
        refined_centers_active = (
            q_initial_active.transpose(0, 1) @ coords_normalized
        ) / (initial_mass.unsqueeze(-1) + eps)
        refined_distance2 = (
            coords_normalized[:, None, :] - refined_centers_active[None, :, :]
        ).square().sum(dim=-1)
        refined_sigma2_active = (
            q_initial_active * refined_distance2
        ).sum(dim=0) / (initial_mass + eps)
        refined_sigma2_active = refined_sigma2_active.clamp_min(
            self.region_min_spatial_scale ** 2
        )
        refined_prototypes_active = refined_prototypes_active.detach()
        refined_centers_active = refined_centers_active.detach()
        refined_sigma2_active = refined_sigma2_active.detach()

        final_score = self._assignment_score(
            patch_embedding,
            refined_prototypes_active,
            coords_normalized,
            refined_centers_active,
            refined_sigma2_active,
        )
        q_soft_active = _region_masked_softmax(
            final_score / self.region_temperature,
            candidate_active,
            dim=1,
            eps=eps,
        )

        # Pad all region-axis tensors to the configured maximum R.
        q_soft_initial = x_vis.new_zeros((patch_count, self.region_num))
        q_soft = x_vis.new_zeros((patch_count, self.region_num))
        candidate_mask = torch.zeros(
            patch_count, self.region_num, dtype=torch.bool, device=x_vis.device
        )
        q_soft_initial[:, :effective_regions] = q_initial_active
        q_soft[:, :effective_regions] = q_soft_active
        candidate_mask[:, :effective_regions] = candidate_active

        active_region_mask = torch.zeros(
            self.region_num, dtype=torch.bool, device=x_vis.device
        )
        active_region_mask[:effective_regions] = True
        seed_indices = torch.full(
            (self.region_num,), -1, dtype=torch.long, device=x_vis.device
        )
        seed_indices[:effective_regions] = seed_active

        hard_ids = torch.argmax(q_soft_active, dim=1)
        hard_ids = hard_ids.clone()
        hard_ids[seed_active] = torch.arange(effective_regions, device=x_vis.device)
        q_hard = F.one_hot(hard_ids, num_classes=self.region_num).to(x_vis.dtype)
        q_st = q_hard - q_soft.detach() + q_soft

        region_counts = q_st.sum(dim=0)
        visual_region = (q_st.transpose(0, 1) @ x_morph) / (
            region_counts.unsqueeze(-1) + eps
        )
        center_region = (q_st.transpose(0, 1) @ coords_normalized) / (
            region_counts.unsqueeze(-1) + eps
        )
        occupancy = region_counts / float(patch_count)

        refined_prototypes = x_vis.new_zeros((self.region_num, self.input_dim))
        refined_centers = x_vis.new_zeros((self.region_num, 2))
        refined_sigma2 = x_vis.new_zeros((self.region_num,))
        refined_prototypes[:effective_regions] = refined_prototypes_active
        refined_centers[:effective_regions] = refined_centers_active
        refined_sigma2[:effective_regions] = refined_sigma2_active

        # Detached regional targets make the cohesion terms optimize the final
        # soft assignments rather than move both assignment and target together.
        patch_unit = F.normalize(x_morph, dim=-1, eps=eps)
        region_unit = F.normalize(visual_region.detach(), dim=-1, eps=eps)
        morph_distance = 1.0 - patch_unit @ region_unit.transpose(0, 1)
        coord_distance = torch.cdist(
            coords_normalized, center_region.detach()
        ).square()
        loss_morph = (q_soft * morph_distance).sum() / float(patch_count)
        loss_coord = (q_soft * coord_distance).sum() / float(patch_count)
        loss_region = loss_morph + self.region_loss_spatial_weight * loss_coord

        return RegionOutput(
            coords_normalized=coords_normalized,
            active_region_mask=active_region_mask,
            seed_indices=seed_indices,
            candidate_mask=candidate_mask,
            q_soft_initial=q_soft_initial,
            q_soft=q_soft,
            q_hard=q_hard,
            q_st=q_st,
            refined_prototypes=refined_prototypes,
            refined_centers=refined_centers,
            refined_sigma2=refined_sigma2,
            region_counts=region_counts,
            occupancy=occupancy,
            visual_region=visual_region,
            center_region=center_region,
            loss_region=loss_region,
            loss_morph=loss_morph,
            loss_coord=loss_coord,
        )

# Full-slide morphology-anchored ST route atlas.
@dataclass(frozen=True)
class RouteAtlasOutput:
    pathway_anchor_prior: Tensor       # [K, C], pi(c | k)
    anchor_region_probability: Tensor  # [C, R], p(r | c)
    pathway_region_prior: Tensor       # [K, R], pi_atlas(r | k)
    pathway_confidence: Tensor         # [K]
    route_cost_bias: Tensor            # [K, R] in [0, 1]
    cost_delta: Tensor                 # [K, R]
    alpha: Tensor                      # scalar
    anchor_valid: Tensor               # [C] bool
    sampled_anchor_valid: Tensor       # [C] bool
    gene_scale: Tensor                 # [G]


class RouteAtlasCalibrator(nn.Module):
    """Convert full-slide pseudo-ST into a conservative UOT route prior.

    The atlas has a fixed morphology axis shared with the coarse stage.  It
    never creates a molecular value or a slide-level shortcut: its only model
    effect is a bounded, confidence-weighted *relative* bias on the existing
    RNA-to-region transport cost.
    """

    def __init__(
        self,
        *,
        st_source_indices: Sequence[Sequence[int]],
        st_slot_valid: Sequence[bool] | Tensor,
        st_gene_count: int,
        anchor_count: int = EXPECTED_MORPHOLOGY_PROTOTYPES,
        temperature: float = 1.0,
        alpha_init: float = 0.02,
        alpha_max: float = 0.20,
        count_scale: float = 16.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.st_gene_count = int(st_gene_count)
        self.anchor_count = int(anchor_count)
        self.temperature = float(temperature)
        self.alpha_max = float(alpha_max)
        self.count_scale = float(count_scale)
        self.eps = float(eps)
        if min(self.st_gene_count, self.anchor_count) <= 0:
            raise ValueError("Route Atlas gene and anchor counts must be positive")
        if self.temperature <= 0 or self.alpha_max <= 0:
            raise ValueError("Route Atlas temperature and alpha_max must be positive")
        if not 0 <= float(alpha_init) < self.alpha_max:
            raise ValueError("Route Atlas alpha_init must lie in [0, alpha_max)")
        if self.count_scale <= 0 or self.eps <= 0:
            raise ValueError("Route Atlas count_scale and eps must be positive")

        slot_valid = torch.as_tensor(st_slot_valid, dtype=torch.bool).reshape(-1)
        if len(st_source_indices) != int(slot_valid.numel()) or not st_source_indices:
            raise ValueError("Route Atlas pathway indices and validity disagree")
        membership = torch.zeros(
            len(st_source_indices), self.st_gene_count, dtype=torch.float32
        )
        for pathway, (indices, valid) in enumerate(
            zip(st_source_indices, slot_valid.tolist())
        ):
            if not valid:
                continue
            unique = tuple(dict.fromkeys(int(index) for index in indices))
            if not unique or any(
                index < 0 or index >= self.st_gene_count for index in unique
            ):
                raise ValueError("Route Atlas received an invalid ST gene index")
            membership[pathway, list(unique)] = 1.0 / float(len(unique))
        pathway_valid = membership.sum(dim=1) > 0
        self.register_buffer("pathway_gene_mean", membership, persistent=True)
        self.register_buffer("pathway_valid", pathway_valid, persistent=True)
        self.alpha_logit = nn.Parameter(bounded_logit(alpha_init, self.alpha_max))

    @property
    def alpha(self) -> Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def _validate_inputs(
        self,
        route_atlas: Tensor,
        route_atlas_counts: Tensor,
        route_atlas_valid: Tensor,
        sampled_anchor_ids: Tensor,
        region_assignment: Tensor,
        active_region_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        expected_atlas = (self.anchor_count, self.st_gene_count)
        if not torch.is_tensor(route_atlas) or tuple(route_atlas.shape) != expected_atlas:
            raise ValueError(f"route_atlas must have shape {expected_atlas}")
        if not route_atlas.is_floating_point():
            raise TypeError("route_atlas must be floating point")
        if (
            not torch.is_tensor(route_atlas_counts)
            or tuple(route_atlas_counts.shape) != (self.anchor_count,)
        ):
            raise ValueError(f"route_atlas_counts must have shape [{self.anchor_count}]")
        if (
            not torch.is_tensor(route_atlas_valid)
            or tuple(route_atlas_valid.shape) != (self.anchor_count,)
        ):
            raise ValueError(f"route_atlas_valid must have shape [{self.anchor_count}]")
        if not torch.is_tensor(sampled_anchor_ids) or sampled_anchor_ids.ndim != 1:
            raise ValueError("sampled_anchor_ids must have shape [N]")
        if (
            not torch.is_tensor(region_assignment)
            or region_assignment.ndim != 2
            or region_assignment.shape[0] != sampled_anchor_ids.shape[0]
        ):
            raise ValueError("region_assignment must have shape [N,R]")
        if (
            not torch.is_tensor(active_region_mask)
            or tuple(active_region_mask.shape) != (region_assignment.shape[1],)
        ):
            raise ValueError("active_region_mask must have shape [R]")

        device = region_assignment.device
        dtype = region_assignment.dtype
        atlas = route_atlas.to(device=device, dtype=dtype)
        counts = route_atlas_counts.to(device=device, dtype=dtype)
        valid = route_atlas_valid.to(device=device, dtype=torch.bool)
        anchor_ids = sampled_anchor_ids.to(device=device, dtype=torch.long)
        active = active_region_mask.to(device=device, dtype=torch.bool)
        for name, values in (("route_atlas", atlas), ("route_atlas_counts", counts)):
            if not bool(torch.isfinite(values).all().detach().item()):
                raise ValueError(f"{name} contains NaN/Inf values")
        if bool((counts < 0).any().detach().item()):
            raise ValueError("route_atlas_counts must be non-negative")
        if not torch.equal(valid, counts > 0):
            raise ValueError("route_atlas_valid must exactly match positive counts")
        if not bool(valid.any().detach().item()):
            raise ValueError("Route Atlas contains no occupied morphology anchor")
        if bool((atlas[~valid] != 0).any().detach().item()):
            raise ValueError("Empty Route Atlas anchors must be exact zeros")
        if anchor_ids.numel() == 0 or bool(
            ((anchor_ids < 0) | (anchor_ids >= self.anchor_count)).any().item()
        ):
            raise ValueError("sampled_anchor_ids contain an invalid anchor ID")
        if not bool(torch.isfinite(region_assignment).all().detach().item()):
            raise ValueError("region_assignment contains NaN/Inf values")
        return atlas, counts, valid, anchor_ids, region_assignment, active

    def forward(
        self,
        *,
        route_atlas: Tensor,
        route_atlas_counts: Tensor,
        route_atlas_valid: Tensor,
        sampled_anchor_ids: Tensor,
        region_assignment: Tensor,
        active_region_mask: Tensor,
    ) -> RouteAtlasOutput:
        atlas, counts, valid, anchor_ids, q_region, active = self._validate_inputs(
            route_atlas,
            route_atlas_counts,
            route_atlas_valid,
            sampled_anchor_ids,
            region_assignment,
            active_region_mask,
        )
        eps = self.eps
        dtype = q_region.dtype
        weights = counts * valid.to(dtype)
        total_count = weights.sum().clamp_min(1.0)
        gene_mean = (weights[:, None] * atlas).sum(dim=0) / total_count
        centered = atlas - gene_mean.unsqueeze(0)
        gene_variance = (
            weights[:, None] * centered.square()
        ).sum(dim=0) / total_count
        gene_scale = torch.sqrt(gene_variance.clamp_min(0.0))
        informative_gene = gene_scale > eps
        standardized = torch.where(
            informative_gene.unsqueeze(0),
            centered / gene_scale.clamp_min(eps).unsqueeze(0),
            torch.zeros_like(centered),
        )
        standardized = standardized * valid[:, None].to(dtype)

        membership = self.pathway_gene_mean.to(device=atlas.device, dtype=dtype)
        pathway_valid = self.pathway_valid.to(device=atlas.device)
        anchor_scores = standardized @ membership.transpose(0, 1)  # [C,K]
        logits = anchor_scores.transpose(0, 1) / self.temperature
        pathway_anchor_mask = pathway_valid[:, None] & valid[None, :]
        pathway_anchor_prior = _transport_masked_softmax(
            logits, pathway_anchor_mask, dim=1, eps=eps
        )

        one_hot = F.one_hot(
            anchor_ids, num_classes=self.anchor_count
        ).to(dtype=dtype)
        sampled_counts = one_hot.sum(dim=0)
        sampled_valid = sampled_counts > 0
        anchor_region = one_hot.transpose(0, 1) @ q_region
        anchor_region = anchor_region / sampled_counts.clamp_min(1.0).unsqueeze(1)
        anchor_region = anchor_region * sampled_valid[:, None].to(dtype)
        anchor_region = anchor_region * active[None, :].to(dtype)

        region_prior = pathway_anchor_prior @ anchor_region
        represented_mass = region_prior.sum(dim=1)
        region_prior = region_prior / represented_mass.clamp_min(eps).unsqueeze(1)
        region_prior = torch.where(
            represented_mass[:, None] > eps,
            region_prior,
            torch.zeros_like(region_prior),
        )
        region_prior = region_prior * active[None, :].to(dtype)

        valid_anchor_count = valid.to(dtype).sum()
        entropy = -(
            pathway_anchor_prior
            * torch.log(pathway_anchor_prior.clamp_min(eps))
        ).sum(dim=1)
        valid_scores = anchor_scores.transpose(0, 1)
        score_max = valid_scores.masked_fill(~valid[None, :], -torch.inf).amax(dim=1)
        score_min = valid_scores.masked_fill(~valid[None, :], torch.inf).amin(dim=1)
        route_informative = (score_max - score_min) > eps
        max_entropy = torch.log(valid_anchor_count.clamp_min(1.0))
        specificity = torch.where(
            valid_anchor_count > 1,
            (1.0 - entropy / max_entropy.clamp_min(eps)).clamp(0.0, 1.0),
            torch.zeros_like(entropy),
        )
        count_quality_by_anchor = counts / (counts + self.count_scale)
        count_quality = (
            pathway_anchor_prior * count_quality_by_anchor.unsqueeze(0)
        ).sum(dim=1)
        sampled_quality_by_anchor = sampled_counts / (
            sampled_counts + self.count_scale
        )
        sampled_quality = (
            pathway_anchor_prior * sampled_quality_by_anchor.unsqueeze(0)
        ).sum(dim=1) / represented_mass.clamp_min(eps)
        sampled_quality = torch.where(
            represented_mass > eps,
            sampled_quality.clamp(0.0, 1.0),
            torch.zeros_like(sampled_quality),
        )
        coverage = (valid_anchor_count / float(self.anchor_count)).clamp(0.0, 1.0)
        confidence = (
            represented_mass.clamp(0.0, 1.0)
            * count_quality.clamp(0.0, 1.0)
            * sampled_quality
            * specificity
            * coverage
            * pathway_valid.to(dtype)
            * route_informative.to(dtype)
        )

        best = region_prior.amax(dim=1, keepdim=True)
        relative = region_prior / best.clamp_min(eps)
        raw_bias = -torch.log(relative.clamp_min(eps))
        raw_bias = raw_bias * active[None, :].to(dtype)
        bias_max = raw_bias.amax(dim=1, keepdim=True)
        route_bias = raw_bias / bias_max.clamp_min(eps)
        route_bias = torch.where(
            (best > eps) & (bias_max > eps),
            route_bias,
            torch.zeros_like(route_bias),
        )
        cost_delta = self.alpha.to(dtype) * confidence.unsqueeze(1) * route_bias
        return RouteAtlasOutput(
            pathway_anchor_prior=pathway_anchor_prior,
            anchor_region_probability=anchor_region,
            pathway_region_prior=region_prior,
            pathway_confidence=confidence,
            route_cost_bias=route_bias,
            cost_delta=cost_delta,
            alpha=self.alpha,
            anchor_valid=valid,
            sampled_anchor_valid=sampled_valid,
            gene_scale=gene_scale,
        )


# Masked region-level KL-unbalanced optimal transport.
@dataclass
class TransportOutput:
    """Region-level molecular transport outputs for one slide."""

    valid_mask: Tensor             # [K, R] bool
    source_valid_mask: Tensor      # [K] bool
    target_valid_mask: Tensor      # [R] bool
    cost: Tensor                   # [K, R] FP32, calibrated when Atlas is active
    local_cost: Tensor             # [K, R] FP32, sampled regional ST only
    atlas_cost_delta: Tensor       # [K, R] FP32 route-atlas adjustment
    pathway_prior: Tensor          # [K] FP32, pi
    source_reference: Tensor       # [K] FP32, a
    total_reference_mass: Tensor   # scalar FP32, A
    region_prior: Tensor           # [R] FP32, rho
    target_reference: Tensor       # [R] FP32, b=A*rho
    transport: Tensor              # [K, R] FP32
    rna_values: Tensor             # [K, value_dim]
    row_mass: Tensor               # [K] FP32
    region_mass: Tensor            # [R] FP32
    total_mass: Tensor             # scalar FP32
    molecular_region: Tensor       # [R, value_dim]
    support: Tensor                # [R] FP32 in [0,1]

    @property
    def pi(self) -> Tensor:
        return self.pathway_prior

    @property
    def a_tr(self) -> Tensor:
        return self.source_reference

    @property
    def source_mass(self) -> Tensor:
        return self.source_reference

    @property
    def A(self) -> Tensor:
        return self.total_reference_mass

    @property
    def rho(self) -> Tensor:
        return self.region_prior

    @property
    def b_tr(self) -> Tensor:
        return self.target_reference


def _transport_masked_softmax(logits: Tensor, mask: Tensor, dim: int, eps: float) -> Tensor:
    mask = mask.to(device=logits.device, dtype=torch.bool)
    valid = mask.any(dim=dim, keepdim=True)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    masked_logits = torch.where(valid, masked_logits, torch.zeros_like(masked_logits))
    probs = F.softmax(masked_logits, dim=dim) * mask.to(logits.dtype)
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    return torch.where(valid, probs, torch.zeros_like(probs))


def _fp32_log_kl_uot(
    cost: Tensor,
    source_reference: Tensor,
    target_reference: Tensor,
    target_prior: Tensor,
    valid_mask: Tensor,
    *,
    iterations: int,
    entropy_epsilon: float,
    tau_source: float,
    tau_target: float,
    mass_epsilon: float,
) -> Tensor:
    """Fixed-iteration generalized Sinkhorn in FP32 log space.

    The zero-plan branches retain a zero-gradient connection to ``cost`` so
    callers can safely differentiate objectives containing a degenerate plan.
    """

    if cost.ndim != 2:
        raise ValueError(f"cost must be [K,R], got {tuple(cost.shape)}")
    cost32 = cost.float()
    source32 = source_reference.float()
    target32 = target_reference.float()
    prior32 = target_prior.float()
    mask = valid_mask.to(device=cost.device, dtype=torch.bool)

    finite_cost = torch.isfinite(cost32)
    positive_source = torch.isfinite(source32) & (source32 > mass_epsilon)
    positive_target = (
        torch.isfinite(target32)
        & torch.isfinite(prior32)
        & (target32 > mass_epsilon)
        & (prior32 > mass_epsilon)
    )
    valid = mask & finite_cost & positive_source[:, None] & positive_target[None, :]
    if not bool(valid.any().detach().item()):
        return cost32 * 0.0

    eps_ot = max(float(entropy_epsilon), float(mass_epsilon))
    theta_source = float(tau_source) / (float(tau_source) + eps_ot)
    theta_target = float(tau_target) / (float(tau_target) + eps_ot)
    log_source = torch.log(source32.clamp_min(mass_epsilon))
    log_target = torch.log(target32.clamp_min(mass_epsilon))
    log_prior = torch.log(prior32.clamp_min(mass_epsilon))
    log_kernel = (
        log_source[:, None]
        + log_prior[None, :]
        - cost32 / eps_ot
    ).masked_fill(~valid, -torch.inf)

    valid_rows = valid.any(dim=1)
    valid_columns = valid.any(dim=0)
    log_u = torch.zeros_like(source32)
    log_v = torch.zeros_like(target32)

    # No early stopping: the fixed execution path is required by Cox replay.
    for _ in range(int(iterations)):
        row_terms = log_kernel + log_v[None, :]
        if bool((~valid_rows).any().detach().item()):
            row_terms = row_terms.clone()
            row_terms[~valid_rows, 0] = 0.0
        row_lse = torch.logsumexp(row_terms, dim=1)
        next_log_u = torch.zeros_like(log_u)
        next_log_u[valid_rows] = theta_source * (
            log_source[valid_rows] - row_lse[valid_rows]
        )
        log_u = next_log_u

        column_terms = log_kernel + log_u[:, None]
        if bool((~valid_columns).any().detach().item()):
            column_terms = column_terms.clone()
            column_terms[0, ~valid_columns] = 0.0
        column_lse = torch.logsumexp(column_terms, dim=0)
        next_log_v = torch.zeros_like(log_v)
        next_log_v[valid_columns] = theta_target * (
            log_target[valid_columns] - column_lse[valid_columns]
        )
        log_v = next_log_v

    log_transport = log_u[:, None] + log_kernel + log_v[None, :]
    transport = torch.exp(log_transport)
    return torch.where(valid, transport, torch.zeros_like(transport))


class RegionKLUOT(nn.Module):
    """Transport measured RNA pathway content directly to fixed region slots."""

    def __init__(
        self,
        pathway_dim: int = 256,
        region_visual_dim: int = 768,
        transport_dim: int = 768,
        value_dim: int = 768,
        transport_epsilon: float = 0.07,
        transport_tau_source: float = 0.5,
        transport_tau_target: float = 0.5,
        transport_sinkhorn_iters: int = 40,
        transport_reliability_beta: float = 1.0,
        region_capacity_occupancy_weight: float = 0.25,
        transport_mass_epsilon: float = 1e-8,
        transport_norm_epsilon: float = 1e-6,
        transport_min_reference_mass: float = 1e-8,
    ) -> None:
        super().__init__()
        self.pathway_dim = int(pathway_dim)
        self.region_visual_dim = int(region_visual_dim)
        self.transport_dim = int(transport_dim)
        self.value_dim = int(value_dim)
        self.transport_epsilon = float(transport_epsilon)
        self.transport_tau_source = float(transport_tau_source)
        self.transport_tau_target = float(transport_tau_target)
        self.transport_sinkhorn_iters = int(transport_sinkhorn_iters)
        self.transport_reliability_beta = float(transport_reliability_beta)
        self.region_capacity_occupancy_weight = float(
            region_capacity_occupancy_weight
        )
        self.transport_mass_epsilon = float(transport_mass_epsilon)
        self.transport_norm_epsilon = float(transport_norm_epsilon)
        self.transport_min_reference_mass = float(transport_min_reference_mass)

        if min(
            self.pathway_dim,
            self.region_visual_dim,
            self.transport_dim,
            self.value_dim,
        ) <= 0:
            raise ValueError("all RegionKLUOT dimensions must be positive")
        if self.transport_epsilon <= 0:
            raise ValueError("transport_epsilon must be positive")
        if self.transport_tau_source <= 0 or self.transport_tau_target <= 0:
            raise ValueError("transport KL penalties must be positive")
        if self.transport_sinkhorn_iters <= 0:
            raise ValueError("transport_sinkhorn_iters must be positive")
        if self.transport_reliability_beta <= 0:
            raise ValueError("transport_reliability_beta must be positive")
        if self.region_capacity_occupancy_weight < 0:
            raise ValueError("region capacity occupancy weight must be non-negative")
        if self.transport_mass_epsilon <= 0 or self.transport_norm_epsilon <= 0:
            raise ValueError("transport epsilons must be positive")
        if self.transport_min_reference_mass < 0:
            raise ValueError("transport_min_reference_mass must be non-negative")

        self.rna_transport_key = nn.Linear(self.pathway_dim, self.transport_dim)
        self.st_transport_query = nn.Linear(self.pathway_dim, self.transport_dim)
        self.rna_transport_value = nn.Linear(self.pathway_dim, self.value_dim)
        self.rna_mass_head = nn.Linear(self.pathway_dim, 1)
        self.region_capacity_head = nn.Linear(self.region_visual_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (
            self.rna_transport_key,
            self.st_transport_query,
            self.rna_transport_value,
            self.rna_mass_head,
            self.region_capacity_head,
        ):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _validate_inputs(
        self,
        rna_pathway: Tensor,
        st_pathway: Tensor,
        rna_pathway_valid: Tensor,
        st_pathway_valid: Tensor,
        st_pathway_reliability: Tensor,
        region_visual: Tensor,
        occupancy: Tensor,
        active_region_mask: Tensor,
    ) -> tuple[int, int]:
        if rna_pathway.ndim != 2 or rna_pathway.shape[1] != self.pathway_dim:
            raise ValueError(
                f"rna_pathway must be [K,{self.pathway_dim}], got "
                f"{tuple(rna_pathway.shape)}"
            )
        pathway_count = int(rna_pathway.shape[0])
        if st_pathway.ndim != 3 or tuple(st_pathway.shape[1:]) != (
            pathway_count,
            self.pathway_dim,
        ):
            raise ValueError(
                f"st_pathway must be [R,{pathway_count},{self.pathway_dim}], "
                f"got {tuple(st_pathway.shape)}"
            )
        region_count = int(st_pathway.shape[0])
        expected_shapes = {
            "rna_pathway_valid": (pathway_count,),
            "st_pathway_valid": (region_count, pathway_count),
            "st_pathway_reliability": (pathway_count,),
            "region_visual": (region_count, self.region_visual_dim),
            "occupancy": (region_count,),
            "active_region_mask": (region_count,),
        }
        values = {
            "rna_pathway_valid": rna_pathway_valid,
            "st_pathway_valid": st_pathway_valid,
            "st_pathway_reliability": st_pathway_reliability,
            "region_visual": region_visual,
            "occupancy": occupancy,
            "active_region_mask": active_region_mask,
        }
        for name, expected in expected_shapes.items():
            value = values[name]
            if not torch.is_tensor(value) or tuple(value.shape) != expected:
                shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
                raise ValueError(f"{name} must have shape {expected}, got {shape}")
        for name, value in (
            ("rna_pathway", rna_pathway),
            ("st_pathway", st_pathway),
            ("st_pathway_reliability", st_pathway_reliability),
            ("region_visual", region_visual),
            ("occupancy", occupancy),
        ):
            if not bool(torch.isfinite(value).all().detach().item()):
                raise ValueError(f"{name} contains NaN/Inf values")
        if bool((occupancy < 0).any().detach().item()):
            raise ValueError("occupancy must be non-negative")
        if bool((st_pathway_reliability < 0).any().detach().item()) or bool(
            (st_pathway_reliability > 1).any().detach().item()
        ):
            raise ValueError("st_pathway_reliability must lie in [0,1]")
        return pathway_count, region_count

    def forward(
        self,
        rna_pathway: Tensor,
        st_pathway: Tensor,
        rna_pathway_valid: Tensor,
        st_pathway_valid: Tensor,
        st_pathway_reliability: Tensor,
        region_visual: Tensor,
        occupancy: Tensor,
        active_region_mask: Tensor,
        route_cost_delta: Optional[Tensor] = None,
    ) -> TransportOutput:
        pathway_count, region_count = self._validate_inputs(
            rna_pathway,
            st_pathway,
            rna_pathway_valid,
            st_pathway_valid,
            st_pathway_reliability,
            region_visual,
            occupancy,
            active_region_mask,
        )
        del pathway_count, region_count
        eps = self.transport_mass_epsilon
        device = rna_pathway.device

        rna_valid = rna_pathway_valid.to(device=device, dtype=torch.bool)
        st_valid = st_pathway_valid.to(device=device, dtype=torch.bool)
        active = active_region_mask.to(device=device, dtype=torch.bool)
        reliability = st_pathway_reliability.to(device=device).float().clamp(0.0, 1.0)

        # Start from K x R biological validity, then prune structurally empty
        # rows before source softmax.  Positive source mass is subsequently used
        # to remove target columns that have no feasible incoming edge.
        structural_mask = (
            rna_valid[:, None]
            & st_valid.transpose(0, 1)
            & active[None, :]
        )
        source_valid = rna_valid & structural_mask.any(dim=1)
        structural_mask = structural_mask & source_valid[:, None]

        rna_logits = self.rna_mass_head(rna_pathway).squeeze(-1).float()
        pathway_prior = _transport_masked_softmax(rna_logits, source_valid, dim=0, eps=eps)
        source_reference = pathway_prior * reliability.pow(
            self.transport_reliability_beta
        )
        total_reference_mass = source_reference.sum()

        positive_source = source_reference > self.transport_min_reference_mass
        valid_mask = structural_mask & positive_source[:, None]
        target_valid = active & valid_mask.any(dim=0)
        valid_mask = valid_mask & target_valid[None, :]
        source_valid_for_transport = source_valid & valid_mask.any(dim=1)

        occupancy32 = occupancy.to(device=device).float()
        capacity_logits = self.region_capacity_head(region_visual).squeeze(-1).float()
        capacity_logits = capacity_logits + self.region_capacity_occupancy_weight * torch.log(
            occupancy32.clamp_min(eps)
        )
        region_prior = _transport_masked_softmax(
            capacity_logits, target_valid, dim=0, eps=eps
        )
        target_reference = total_reference_mass * region_prior

        rna_key = F.normalize(
            self.rna_transport_key(rna_pathway).float(),
            dim=-1,
            eps=self.transport_norm_epsilon,
        )
        st_query = F.normalize(
            self.st_transport_query(st_pathway).float(),
            dim=-1,
            eps=self.transport_norm_epsilon,
        )
        cosine = (st_query * rna_key.unsqueeze(0)).sum(dim=-1).transpose(0, 1)
        local_cost = ((1.0 - cosine) * 0.5).clamp(0.0, 1.0)
        if route_cost_delta is None:
            atlas_cost_delta = torch.zeros_like(local_cost)
        else:
            if (
                not torch.is_tensor(route_cost_delta)
                or tuple(route_cost_delta.shape) != tuple(local_cost.shape)
            ):
                raise ValueError(
                    "route_cost_delta must match the [K,R] transport cost"
                )
            atlas_cost_delta = route_cost_delta.to(
                device=local_cost.device, dtype=local_cost.dtype
            )
            if not bool(torch.isfinite(atlas_cost_delta).all().detach().item()):
                raise ValueError("route_cost_delta contains NaN/Inf values")
            if bool((atlas_cost_delta < 0).any().detach().item()):
                raise ValueError("route_cost_delta must be non-negative")
        cost = local_cost + atlas_cost_delta

        degenerate = (
            not bool(valid_mask.any().detach().item())
            or bool(
                (
                    total_reference_mass.detach()
                    <= self.transport_min_reference_mass
                ).item()
            )
        )
        if degenerate:
            transport = cost * 0.0
        else:
            transport = _fp32_log_kl_uot(
                cost=cost,
                source_reference=source_reference,
                target_reference=target_reference,
                target_prior=region_prior,
                valid_mask=valid_mask,
                iterations=self.transport_sinkhorn_iters,
                entropy_epsilon=self.transport_epsilon,
                tau_source=self.transport_tau_source,
                tau_target=self.transport_tau_target,
                mass_epsilon=self.transport_mass_epsilon,
            )

        rna_values = self.rna_transport_value(rna_pathway)
        row_mass = transport.sum(dim=1)
        region_mass = transport.sum(dim=0)
        total_mass = region_mass.sum()
        molecular_fp32 = (transport.transpose(0, 1) @ rna_values.float()) / (
            region_mass.unsqueeze(-1) + eps
        )
        molecular_region = molecular_fp32.to(dtype=rna_values.dtype)
        support = (
            region_mass / (target_reference + eps)
        ).clamp(min=0.0, max=1.0)
        support = support * active.to(dtype=support.dtype)

        return TransportOutput(
            valid_mask=valid_mask,
            source_valid_mask=source_valid_for_transport,
            target_valid_mask=target_valid,
            cost=cost,
            local_cost=local_cost,
            atlas_cost_delta=atlas_cost_delta,
            pathway_prior=pathway_prior,
            source_reference=source_reference,
            total_reference_mass=total_reference_mass,
            region_prior=region_prior,
            target_reference=target_reference,
            transport=transport,
            rna_values=rna_values,
            row_mass=row_mass,
            region_mass=region_mass,
            total_mass=total_mass,
            molecular_region=molecular_region,
            support=support,
        )

# Regional memory and TITAN reader.
@dataclass
class RegionMemoryOutput:
    keys: torch.Tensor
    values: torch.Tensor
    support: torch.Tensor
    active_mask: torch.Tensor


@dataclass
class RegionInteractionState:
    """Slide-local dynamic keys exchanged between TITAN callbacks.

    RNA-derived ``RegionMemoryOutput.values`` deliberately do not live here,
    so a contextual writeback cannot replace molecular evidence.
    """

    keys: torch.Tensor
    context: Optional[torch.Tensor] = None
    updates: int = 0


class RegionMemory(nn.Module):
    def __init__(
        self,
        *,
        visual_dim: int = 768,
        memory_dim: int = 768,
        coord_num_freqs: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.coord_num_freqs = int(coord_num_freqs)
        self.eps = float(eps)
        coord_dim = 2 + 4 * self.coord_num_freqs
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.key_projection = nn.Linear(visual_dim + coord_dim + 1, memory_dim)
        self.value_norm = nn.LayerNorm(memory_dim)
        self.value_projection = nn.Linear(memory_dim, memory_dim)
        init_new_module(self)

    def _coordinate_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        features = [coords]
        for index in range(self.coord_num_freqs):
            frequency = float(2**index) * math.pi
            features.extend(
                [torch.sin(frequency * coords), torch.cos(frequency * coords)]
            )
        return torch.cat(features, dim=-1)

    def adapt_values(self, molecular_region: torch.Tensor) -> torch.Tensor:
        return self.value_projection(self.value_norm(molecular_region))

    def forward(
        self,
        region,
        transport,
    ) -> RegionMemoryOutput:
        active = region.active_mask.to(dtype=torch.bool)
        active_f = active.unsqueeze(-1).to(region.visual.dtype)
        key_input = torch.cat(
            [
                self.visual_norm(region.visual),
                self._coordinate_encoding(region.center),
                torch.log(region.occupancy.clamp_min(self.eps)).unsqueeze(-1),
            ],
            dim=-1,
        )
        keys = self.key_projection(key_input) * active_f
        values = self.adapt_values(
            transport.molecular_region.to(region.visual.dtype)
        ) * active_f
        support = (
            transport.support.to(region.visual.dtype)
            * active.to(region.visual.dtype)
        )
        return RegionMemoryOutput(
            keys=keys,
            values=values,
            support=support,
            active_mask=active,
        )

class TitanRegionMemoryReader(nn.Module):
    """Create a deterministic dynamic callback for selected TITAN blocks."""

    def __init__(
        self,
        *,
        hidden_dim: int = 768,
        inject_layers=(2, 4),
        assignment_log_weight: float = 1.0,
        alpha_max: float = 0.1,
        alpha_init: float = 0.05,
        eps: float = 1e-6,
        mass_eps: float = 1e-8,
        min_confidence_sum: float = 1e-8,
        align_cosine_margin: float = 0.0,
        align_min_total_mass: float = 1e-8,
        align_min_region_mass: float = 1e-8,
        align_min_region_occupancy: float = 1e-3,
        align_min_anchor_norm: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.inject_layers = tuple(sorted({int(layer) for layer in inject_layers}))
        self.assignment_log_weight = float(assignment_log_weight)
        self.alpha_max = float(alpha_max)
        self.eps = float(eps)
        self.mass_eps = float(mass_eps)
        self.min_confidence_sum = float(min_confidence_sum)
        self.align_cosine_margin = float(align_cosine_margin)
        self.align_min_total_mass = float(align_min_total_mass)
        self.align_min_region_mass = float(align_min_region_mass)
        self.align_min_region_occupancy = float(align_min_region_occupancy)
        self.align_min_anchor_norm = float(align_min_anchor_norm)
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.local_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2 + 1),
            nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.alpha_logit = nn.Parameter(bounded_logit(alpha_init, self.alpha_max))
        self.layer_logits = nn.Parameter(
            torch.zeros(max(len(self.inject_layers), 1))
        )
        init_new_module(self.query_projection)
        init_new_module(self.local_gate)

    def make_callback(
        self,
        *,
        memory: RegionMemoryOutput,
        transport,
        value_adapter: Callable[[torch.Tensor], torch.Tensor],
        records: Dict[str, Any],
        interaction_state: RegionInteractionState,
    ):
        if not self.inject_layers:
            return None
        active_layers = set(self.inject_layers)
        first_layer = self.inject_layers[0]
        allocation = F.softmax(
            self.layer_logits[: len(self.inject_layers)], dim=0
        )
        total_strength = self.alpha_max * torch.sigmoid(self.alpha_logit)
        strengths = {
            layer: total_strength * allocation[position]
            for position, layer in enumerate(self.inject_layers)
        }
        region_count = memory.keys.shape[0]

        def callback(layer_id, h_pre, auxiliary_tokens):
            layer_id = int(layer_id)
            if layer_id not in active_layers:
                return None
            if auxiliary_tokens is None or auxiliary_tokens.ndim != 3:
                raise ValueError("STARPath callback requires [B,L,C] auxiliary tokens")
            if h_pre.ndim != 3 or h_pre.shape[:2] != auxiliary_tokens.shape[:2]:
                raise ValueError("STARPath callback context is not aligned with TITAN")
            if auxiliary_tokens.shape[-1] != 2 * region_count:
                raise ValueError(
                    f"Expected {2 * region_count} callback channels, got "
                    f"{auxiliary_tokens.shape[-1]}"
                )

            q_raw = auxiliary_tokens[..., :region_count].to(dtype=h_pre.dtype)
            candidate = auxiliary_tokens[..., region_count:] > 0.5
            candidate = candidate & memory.active_mask.view(1, 1, -1).to(
                candidate.device
            )
            q_sum = q_raw.sum(dim=-1, keepdim=True)
            q_prior = torch.where(
                q_sum > 0,
                q_raw / q_sum.clamp_min(self.eps),
                torch.zeros_like(q_raw),
            )
            memory_keys = interaction_state.keys
            if tuple(memory_keys.shape) != tuple(memory.keys.shape):
                raise ValueError("Dynamic region-memory keys changed shape")
            query = self.query_projection(h_pre)
            logits = torch.einsum(
                "bld,rd->blr",
                query,
                memory_keys.to(device=h_pre.device, dtype=query.dtype),
            ) / math.sqrt(float(query.shape[-1]))
            logits = logits + self.assignment_log_weight * torch.log(
                q_prior.clamp_min(self.eps)
            )
            has_candidate = candidate.any(dim=-1, keepdim=True)
            logits = logits.masked_fill(~candidate, torch.finfo(logits.dtype).min)
            logits = torch.where(has_candidate, logits, torch.zeros_like(logits))
            attention = F.softmax(logits, dim=-1)
            attention = attention * candidate.to(attention.dtype)
            attention = torch.where(
                has_candidate,
                attention / attention.sum(dim=-1, keepdim=True).clamp_min(self.eps),
                torch.zeros_like(attention),
            )

            region_support = memory.support.to(
                device=h_pre.device, dtype=attention.dtype
            )
            # Molecular values stay immutable; TITAN contextualizes lookup keys.
            memory_values = memory.values
            patch_support = torch.einsum("blr,r->bl", attention, region_support)
            weighted_attention = attention * region_support.view(1, 1, -1)
            readout = torch.einsum(
                "blr,rd->bld",
                weighted_attention,
                memory_values.to(device=h_pre.device, dtype=h_pre.dtype),
            ) / (patch_support.unsqueeze(-1) + self.mass_eps)

            rms_weight = patch_support.detach().float()
            weight_sum = rms_weight.sum()
            if bool((weight_sum.detach() <= self.min_confidence_sum).item()):
                scaled_readout = torch.zeros_like(readout)
                rms_hidden = h_pre.float().pow(2).mean().sqrt()
                rms_ratio = h_pre.new_zeros(())
            else:
                dimension = float(h_pre.shape[-1])
                readout_sq = readout.float().pow(2).sum(dim=-1)
                hidden_sq = h_pre.float().pow(2).sum(dim=-1)
                rms_readout = torch.sqrt(
                    (rms_weight * readout_sq).sum()
                    / (dimension * (weight_sum + self.eps))
                )
                rms_hidden = torch.sqrt(
                    (rms_weight * hidden_sq).sum()
                    / (dimension * (weight_sum + self.eps))
                )
                scaled_readout = readout / (
                    rms_readout.to(readout.dtype) + self.eps
                )
                scaled_readout = scaled_readout * rms_hidden.detach().to(readout.dtype)

            gate = torch.sigmoid(
                self.local_gate(
                    torch.cat(
                        [h_pre, scaled_readout, patch_support.unsqueeze(-1)], dim=-1
                    )
                )
            )
            delta = (
                strengths[layer_id].to(h_pre.dtype)
                * patch_support.unsqueeze(-1)
                * gate
                * scaled_readout
            )
            if bool((weight_sum.detach() > self.min_confidence_sum).item()):
                dimension = float(h_pre.shape[-1])
                rms_delta = torch.sqrt(
                    (rms_weight * delta.float().pow(2).sum(dim=-1)).sum()
                    / (dimension * (weight_sum + self.eps))
                )
                rms_ratio = rms_delta / (rms_hidden.detach() + self.eps)

            if layer_id == first_layer and "alignment_loss" not in records:
                q_align = q_prior.detach()
                occupancy = q_align.sum(dim=(0, 1))
                anchor = torch.einsum("blr,bld->rd", q_align, h_pre) / (
                    occupancy.unsqueeze(-1) + self.eps
                )
                anchor = anchor.detach()
                region_mass = transport.region_mass.detach().to(occupancy.device)
                active = (
                    (region_mass > self.align_min_region_mass)
                    & (occupancy > self.align_min_region_occupancy)
                    & (anchor.norm(dim=-1) > self.align_min_anchor_norm)
                )
                enough_mass = bool(
                    (transport.transport.detach().sum() > self.align_min_total_mass).item()
                )
                if enough_mass and bool(active.any().item()):
                    adapted = value_adapter(
                        transport.molecular_region.detach()[active].to(h_pre.dtype)
                    )
                    cosine = F.cosine_similarity(
                        adapted, anchor[active], dim=-1, eps=self.eps
                    )
                    error = torch.relu(self.align_cosine_margin - cosine).pow(2)
                    weights = region_mass[active].to(error.dtype)
                    alignment = (weights * error).sum() / weights.sum().clamp_min(
                        self.eps
                    )
                else:
                    alignment = h_pre.new_zeros(())
                records["alignment_loss"] = alignment
                records["alignment_active_mask"] = active.detach()

            layer_record = {
                "attention": attention.detach(),
                "support": patch_support.detach(),
                "gate": gate.detach(),
                "delta": delta.detach(),
                "weighted_rms_ratio": rms_ratio.detach(),
                "strength": strengths[layer_id].detach(),
                "memory_keys": memory_keys.detach().clone(),
                "memory_values": memory_values.detach().clone(),
            }
            records.setdefault("layers", {})[layer_id] = layer_record
            return delta

        records["total_strength"] = total_strength.detach()
        records["layer_allocation"] = allocation.detach()
        return callback


class DynamicRegionMemoryWriter(nn.Module):
    """Write contextual TITAN evidence back into region lookup keys only."""

    def __init__(
        self,
        *,
        hidden_dim: int = 768,
        alpha_init: float = 0.05,
        alpha_max: float = 0.10,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.alpha_max = float(alpha_max)
        self.eps = float(eps)
        if self.hidden_dim <= 0 or self.alpha_max <= 0 or self.eps <= 0:
            raise ValueError("Dynamic writer dimensions and bounds must be positive")
        if not 0 <= float(alpha_init) < self.alpha_max:
            raise ValueError("dynamic alpha_init must lie in [0, alpha_max)")
        self.context_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.key_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2 + 1),
            nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.alpha_logit = nn.Parameter(bounded_logit(alpha_init, self.alpha_max))
        init_new_module(self.context_projection)
        init_new_module(self.key_gate)

    @property
    def alpha(self) -> Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def make_post_callback(
        self,
        *,
        write_layer: int,
        memory: RegionMemoryOutput,
        interaction_state: RegionInteractionState,
        records: Dict[str, Any],
    ):
        write_layer = int(write_layer)
        region_count = int(memory.keys.shape[0])
        if tuple(interaction_state.keys.shape) != tuple(memory.keys.shape):
            raise ValueError("Dynamic interaction keys do not match region memory")

        def post_callback(layer_id, h_post, auxiliary_tokens):
            if int(layer_id) != write_layer:
                return None
            if interaction_state.updates != 0:
                raise RuntimeError("Dynamic region memory was written more than once")
            if auxiliary_tokens is None or auxiliary_tokens.ndim != 3:
                raise ValueError("Dynamic writeback requires [B,L,C] auxiliary tokens")
            if h_post.ndim != 3 or h_post.shape[:2] != auxiliary_tokens.shape[:2]:
                raise ValueError("Dynamic writeback context is not aligned with TITAN")
            if auxiliary_tokens.shape[-1] != 2 * region_count:
                raise ValueError("Dynamic writeback received the wrong region width")

            q_region = auxiliary_tokens[..., :region_count].to(dtype=h_post.dtype)
            q_region = q_region.clamp_min(0.0)
            q_region = q_region * memory.active_mask.view(1, 1, -1).to(
                device=q_region.device, dtype=q_region.dtype
            )
            region_weight = q_region.sum(dim=(0, 1))
            context = torch.einsum("blr,bld->rd", q_region, h_post)
            context = context / region_weight.clamp_min(self.eps).unsqueeze(1)
            context_valid = memory.active_mask.to(context.device) & (
                region_weight > self.eps
            )
            context = context * context_valid.unsqueeze(1).to(context.dtype)

            base_keys = interaction_state.keys.to(
                device=h_post.device, dtype=h_post.dtype
            )
            projected = self.context_projection(context)
            base_rms = base_keys.float().square().mean(dim=1, keepdim=True).sqrt()
            projected_rms = projected.float().square().mean(dim=1, keepdim=True).sqrt()
            scaled_context = projected / (
                projected_rms.to(projected.dtype) + self.eps
            )
            scaled_context = scaled_context * base_rms.detach().to(projected.dtype)
            support = memory.support.to(device=h_post.device, dtype=h_post.dtype)
            gate = torch.sigmoid(
                self.key_gate(
                    torch.cat(
                        [base_keys, scaled_context, support.unsqueeze(1)], dim=1
                    )
                )
            )
            update_mask = context_valid.to(h_post.dtype) * support
            delta = (
                self.alpha.to(h_post.dtype)
                * update_mask.unsqueeze(1)
                * gate
                * scaled_context
            )
            updated_keys = base_keys + delta
            updated_keys = updated_keys * memory.active_mask.unsqueeze(1).to(
                device=updated_keys.device, dtype=updated_keys.dtype
            )
            interaction_state.keys = updated_keys
            interaction_state.context = context
            interaction_state.updates += 1
            records["dynamic_writeback"] = {
                "layer": write_layer,
                "context": context.detach(),
                "region_weight": region_weight.detach(),
                "gate": gate.detach(),
                "delta": delta.detach(),
                "base_keys": base_keys.detach().clone(),
                "updated_keys": updated_keys.detach().clone(),
                "alpha": self.alpha.detach(),
            }
            return None

        return post_callback

# Shared fine-stage runtime owned by the joint facade below.
class _FineSTARPath(nn.Module):
    """Shared region/UOT/TITAN implementation for the public joint model."""

    def __init__(
        self,
        input_dim: int = 768,
        n_classes: int = 4,
        mode: str = "classification",
        ds_num: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        if self.input_dim != 768:
            raise ValueError(
                "STARPath/TITAN requires 768-dimensional patch features, "
                f"got {self.input_dim}"
            )
        if ds_num is not None:
            raise ValueError(
                "STARPath sampling belongs in the aligned WSI/ST dataset; "
                "ds_num is unsupported"
            )
        self.mode = str(mode)
        self.rna_gene_seq = normalise_gene_axis(
            kwargs.get("rna_gene_seq"), "rna"
        )
        self.st_gene_seq = normalise_gene_axis(
            kwargs.get("st_gene_seq"), "st"
        )

        pathway_type = str(kwargs.get("pathway_type", "hallmarks")).strip().lower()
        if pathway_type not in {"hallmark", "hallmarks"}:
            raise ValueError(
                "STARPath uses exactly the 50 Hallmark pathways; "
                f"received pathway_type={pathway_type!r}"
            )
        self.pathway_type = "hallmarks"
        signature_path = kwargs.get(
            "pathway_signature_path", None
        )
        if signature_path is None:
            signature_path = os.path.join(
                os.path.dirname(__file__), "..", "data_csvs", "rna", "metadata",
                "hallmarks_signatures.csv",
            )
        self.pathway_signature_path = absolute_path(signature_path)
        if not os.path.isfile(self.pathway_signature_path):
            raise FileNotFoundError(
                "STARPath Hallmark signature file does not exist: "
                f"{self.pathway_signature_path}"
            )

        self.pathway_dim = int(kwargs.get("pathway_dim", 256))
        self.pathway_encoder = SurvPathPathwayEncoder(
            signature_path=self.pathway_signature_path,
            rna_gene_seq=self.rna_gene_seq,
            st_gene_seq=self.st_gene_seq,
            token_dim=self.pathway_dim,
            hidden_dim=int(kwargs.get("pathway_hidden_dim", 256)),
            dropout=float(kwargs.get("pathway_dropout", 0.25)),
            min_rna_genes=int(kwargs.get("min_pathway_genes", 1)),
            min_st_genes=int(kwargs.get("min_pathway_genes", 1)),
        )
        self.pathway_names = list(self.pathway_encoder.pathway_names)
        if len(self.pathway_names) != 50:
            raise ValueError(
                "STARPath requires 50 Hallmark pathways, but loaded "
                f"{len(self.pathway_names)}"
            )

        self.region_num = int(kwargs.get("region_num", 12))
        self.region_constructor = MorphologySpaceRegionConstructor(
            input_dim=self.input_dim,
            region_num=self.region_num,
            assignment_dim=int(kwargs.get("region_assignment_dim", 256)),
            region_seed_knn=int(kwargs.get("region_seed_knn", 8)),
            region_candidate_topk=int(kwargs.get("region_candidate_topk", 3)),
            region_temperature=float(kwargs.get("region_temperature", 0.2)),
            region_score_spatial_weight=float(
                kwargs.get("region_score_spatial_weight", 1.0)
            ),
            region_min_spatial_scale=float(
                kwargs.get("region_min_spatial_scale", 0.05)
            ),
            region_loss_spatial_weight=float(
                kwargs.get("region_loss_spatial_weight", 1.0)
            ),
            region_norm_epsilon=float(
                kwargs.get("region_norm_epsilon", 1e-6)
            ),
        )

        self.route_atlas = (
            (RouteAtlasCalibrator(
                st_source_indices=self.pathway_encoder.st_source_indices,
                st_slot_valid=self.pathway_encoder.st_slot_valid,
                st_gene_count=len(self.st_gene_seq),
                temperature=float(
                    kwargs.get(
                        "atlas_temperature",
                        1.0,
                    )
                ),
                alpha_init=float(
                    kwargs.get(
                        "atlas_alpha_init",
                        0.02,
                    )
                ),
                alpha_max=float(
                    kwargs.get(
                        "atlas_alpha_max",
                        0.20,
                    )
                ),
                count_scale=float(
                    kwargs.get(
                        "atlas_count_scale",
                        16.0,
                    )
                ),
                eps=float(kwargs.get("transport_mass_epsilon", 1e-8)),
            ))
        )

        self.transport = RegionKLUOT(
            pathway_dim=self.pathway_dim,
            region_visual_dim=self.input_dim,
            value_dim=768,
            transport_epsilon=float(kwargs.get("transport_epsilon", 0.07)),
            transport_tau_source=float(
                kwargs.get("transport_tau_source", 0.5)
            ),
            transport_tau_target=float(
                kwargs.get("transport_tau_target", 0.5)
            ),
            transport_sinkhorn_iters=int(
                kwargs.get("transport_sinkhorn_iters", 40)
            ),
            transport_reliability_beta=float(
                kwargs.get("transport_reliability_beta", 1.0)
            ),
            region_capacity_occupancy_weight=float(
                kwargs.get(
                    "transport_occupancy_log_weight",
                    0.25,
                )
            ),
            transport_min_reference_mass=float(
                kwargs.get("transport_min_reference_mass", 1e-8)
            ),
            transport_mass_epsilon=float(
                kwargs.get("transport_mass_epsilon", 1e-8)
            ),
        )

        self.model_path = absolute_path(kwargs.get("model_path") or DEFAULT_TITAN_MODEL_PATH)
        if os.path.basename(self.model_path) != "TITAN_STARPath":
            raise ValueError("STARPath resources must be loaded from a TITAN_STARPath directory")
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"STARPath TITAN directory does not exist: {self.model_path}")
        self.model = _load_titan(self.model_path)
        encoder_method = getattr(
            self.model, "encode_slide_from_patch_features", None
        )
        if not callable(encoder_method):
            raise AttributeError("STARPath private TITAN lacks slide encoding")
        callback_parameters = inspect.signature(encoder_method).parameters
        required = {
            "inject_layers",
            "inject_callback",
            "inject_callback_context",
            "post_block_callback",
        }
        if not required.issubset(callback_parameters):
            raise RuntimeError(
                "STARPath must load TITAN_STARPath with the pre-block callback "
                f"API; missing {sorted(required.difference(callback_parameters))}"
            )

        requested_layers = kwargs.get("titan_inject_layers")
        if requested_layers is None:
            requested_layers = [2, 4]
        self.titan_inject_layers = parse_injection_layers(requested_layers)
        if not self.titan_inject_layers:
            raise ValueError("titan_inject_layers must not be empty")
        if self.titan_inject_layers[-1] == 0:
            raise ValueError("The final TITAN injection layer must be greater than zero so dynamic memory can be written by a preceding block")
        self.titan_vision_depth = int(self.model.config.vision_config.depth)
        negative_layers = [
            layer for layer in self.titan_inject_layers if layer < 0
        ]
        if negative_layers:
            raise ValueError(
                f"TITAN injection layers {negative_layers} must be non-negative"
            )
        invalid_layers = [
            layer for layer in self.titan_inject_layers
            if layer >= self.titan_vision_depth
        ]
        if invalid_layers:
            raise ValueError(
                f"TITAN injection layers {invalid_layers} are outside "
                f"[0,{self.titan_vision_depth})"
            )

        self.titan_trainable_layers = parse_trainable_layers(
            kwargs.get("titan_trainable_layers", (2, 3, 4, 5))
        )
        negative_trainable = [
            layer for layer in self.titan_trainable_layers if layer < 0
        ]
        if negative_trainable:
            raise ValueError(
                f"TITAN trainable layers {negative_trainable} must be non-negative"
            )
        invalid_trainable = [
            layer for layer in self.titan_trainable_layers
            if layer >= self.titan_vision_depth
        ]
        if invalid_trainable:
            raise ValueError(
                f"TITAN trainable layers {invalid_trainable} are outside "
                f"[0,{self.titan_vision_depth})"
            )
        self._configure_titan_trainability()
        self.memory_builder = RegionMemory(
            visual_dim=768,
            memory_dim=768,
            coord_num_freqs=int(kwargs.get("region_coord_num_freqs", 4)),
            eps=float(kwargs.get("region_norm_epsilon", 1e-6)),
        )
        self.memory_reader = TitanRegionMemoryReader(
            hidden_dim=768,
            inject_layers=self.titan_inject_layers,
            assignment_log_weight=float(
                kwargs.get("region_assignment_log_weight", 1.0)
            ),
            alpha_max=float(
                kwargs.get(
                    "inject_alpha_max",
                    0.1,
                )
            ),
            alpha_init=float(
                kwargs.get(
                    "inject_alpha_init",
                    0.05,
                )
            ),
            eps=float(kwargs.get("inject_rms_epsilon", 1e-6)),
            mass_eps=float(kwargs.get("transport_mass_epsilon", 1e-8)),
            min_confidence_sum=float(
                kwargs.get("inject_min_confidence_sum", 1e-8)
            ),
            align_cosine_margin=float(kwargs.get("align_cosine_margin", 0.0)),
            align_min_total_mass=float(
                kwargs.get("align_min_total_mass", 1e-8)
            ),
            align_min_region_mass=float(
                kwargs.get("align_min_region_mass", 1e-8)
            ),
            align_min_region_occupancy=float(
                kwargs.get("align_min_region_occupancy", 1e-3)
            ),
            align_min_anchor_norm=float(
                kwargs.get("align_min_anchor_norm", 1e-6)
            ),
        )
        self.dynamic_memory_writer = (
            (DynamicRegionMemoryWriter(
                hidden_dim=768,
                alpha_init=float(
                    kwargs.get(
                        "dynamic_alpha_init",
                        0.05,
                    )
                ),
                alpha_max=float(
                    kwargs.get(
                        "dynamic_alpha_max",
                        0.10,
                    )
                ),
                eps=float(kwargs.get("inject_rms_epsilon", 1e-6)),
            ))
        )
        # Preserve the normalization of the former state-off path without any
        # codebook/state parameters or computation.
        self.slide_final_norm = nn.LayerNorm(768)
        init_new_module(self.slide_final_norm)

        self.classifier = nn.Linear(768, int(n_classes))
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)
        self.loss_w_region = nonnegative_weight(
            kwargs.get("loss_w_region", 0.05), "loss_w_region"
        )
        self.loss_w_align = nonnegative_weight(
            kwargs.get("loss_w_align", 0.05), "loss_w_align"
        )

    def _configure_titan_trainability(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        blocks = getattr(
            getattr(self.model.vision_encoder, "blocks", None),
            "modules_list",
            None,
        )
        if blocks is None or len(blocks) != self.titan_vision_depth:
            raise AttributeError(
                "STARPath requires TITAN vision_encoder.blocks.modules_list"
            )
        for layer in self.titan_trainable_layers:
            for parameter in blocks[layer].parameters():
                parameter.requires_grad = True
        # The public layer list is the source of truth; it is independent
        # from the layers receiving molecular injection.
        self.titan_trainable_layers = list(self.titan_trainable_layers)
        trainable = set(self.titan_trainable_layers)
        self.titan_frozen_layers = [
            layer
            for layer in range(self.titan_vision_depth)
            if layer not in trainable
        ]
        self.titan_trainable_parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        self.titan_total_parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        if not self.titan_trainable_layers:
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.titan_trainable_layers:
            self.model.eval()
        return self

    def _regional_st(self, region, st: torch.Tensor) -> torch.Tensor:
        regional = region.q_st.transpose(0, 1) @ st
        regional = regional / region.region_counts.unsqueeze(-1).clamp_min(
            self.region_constructor.region_norm_epsilon
        )
        return regional * region.active_region_mask.unsqueeze(-1).to(
            regional.dtype
        )

    def _encode_titan(
        self,
        x1: torch.Tensor,
        coords: torch.Tensor,
        patch_size: int,
        callback,
        callback_context: torch.Tensor,
        post_block_callback=None,
    ) -> torch.Tensor:
        embedding = self.model.encode_slide_from_patch_features(
            x1.unsqueeze(0),
            coords.unsqueeze(0),
            np.int64(patch_size),
            inject_layers=(
                self.titan_inject_layers if callback is not None else None
            ),
            inject_callback=callback,
            inject_callback_context=(
                callback_context.unsqueeze(0) if callback is not None else None
            ),
            post_block_callback=post_block_callback,
        )
        if not torch.is_tensor(embedding):
            raise TypeError("TITAN slide encoder returned a non-tensor")
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.ndim != 2 or embedding.shape[-1] != 768:
            raise ValueError(
                f"TITAN returned an invalid slide shape {tuple(embedding.shape)}"
            )
        embedding = embedding @ self.model.vision_encoder.proj
        return F.normalize(embedding, dim=-1)

    def classify_patient(self, slide_features):
        if isinstance(slide_features, (list, tuple)):
            if not slide_features:
                raise ValueError("classify_patient received no slides")
            slide_features = torch.cat(
                [feature.reshape(1, -1) for feature in slide_features], dim=0
            )
        elif torch.is_tensor(slide_features):
            if slide_features.ndim == 1:
                slide_features = slide_features.unsqueeze(0)
            elif slide_features.ndim == 3 and slide_features.shape[1] == 1:
                slide_features = slide_features.squeeze(1)
        else:
            raise TypeError("slide_features must be a tensor or sequence")
        if slide_features.ndim != 2 or slide_features.shape[-1] != 768:
            raise ValueError("slide_features must have shape [S,768]")
        logits = self.classifier(slide_features.mean(dim=0, keepdim=True))
        return logits, torch.argmax(logits, dim=-1), torch.sigmoid(logits)

    def _record_results(
        self,
        results: Dict[str, Any],
        *,
        slide_embedding,
        region,
        transport,
        memory,
        atlas_output,
        alignment_loss,
        callback_records,
    ) -> None:
        results.update(
            {
                "slide_feat": slide_embedding,
                "pathway_names": list(self.pathway_names),
                "region_active_mask": region.active_mask.detach(),
                "region_seed_indices": region.seed_indices.detach(),
                "region_candidate_mask": region.candidate_mask.detach(),
                "region_q_soft_initial": region.q_soft_initial.detach(),
                "region_q_soft": region.q_soft.detach(),
                "region_q_hard": region.q_hard.detach(),
                "region_occupancy": region.occupancy.detach(),
                "region_centers": region.center.detach(),
                "region_loss": region.loss_region,
                "region_morph_loss": region.loss_morph,
                "region_coord_loss": region.loss_coord,
                "transport_valid_mask": transport.valid_mask.detach(),
                "transport_cost": transport.cost.detach(),
                "transport_local_cost": transport.local_cost.detach(),
                "transport_atlas_cost_delta": transport.atlas_cost_delta.detach(),
                "transport_plan": transport.transport.detach(),
                "transport_source_mass": transport.source_reference.detach(),
                "transport_target_capacity": transport.rho.detach(),
                "transport_region_mass": transport.region_mass.detach(),
                "transport_region_support": transport.support.detach(),
                "region_memory_keys": memory.keys.detach(),
                "region_memory_values": memory.values.detach(),
                "alignment_loss": alignment_loss,
                "titan_inject_layers": list(self.titan_inject_layers),
                "titan_trainable_layers": list(self.titan_trainable_layers),
                "titan_frozen_layers": list(self.titan_frozen_layers),
                "titan_trainable_parameter_count": self.titan_trainable_parameter_count,
                "titan_total_parameter_count": self.titan_total_parameter_count,
                "titan_model_path": self.model_path,
            }
        )
        results.update(
            {
                "route_atlas_anchor_prior": (
                    atlas_output.pathway_anchor_prior.detach()
                ),
                "route_atlas_anchor_region_probability": (
                    atlas_output.anchor_region_probability.detach()
                ),
                "route_atlas_region_prior": (
                    atlas_output.pathway_region_prior.detach()
                ),
                "route_atlas_confidence": (
                    atlas_output.pathway_confidence.detach()
                ),
                "route_atlas_cost_bias": atlas_output.route_cost_bias.detach(),
                "route_atlas_alpha": atlas_output.alpha.detach(),
                "route_atlas_anchor_valid": atlas_output.anchor_valid.detach(),
                "route_atlas_sampled_anchor_valid": (
                    atlas_output.sampled_anchor_valid.detach()
                ),
            }
        )
        if "total_strength" in callback_records:
            results["titan_inject_scale"] = callback_records[
                "total_strength"
            ]
            results["titan_layer_allocation"] = callback_records[
                "layer_allocation"
            ]
        for layer_id, record in callback_records.get("layers", {}).items():
            results[f"titan_region_attention_l{layer_id}"] = record["attention"]
            results[f"titan_region_support_l{layer_id}"] = record["support"]
            results[f"titan_region_gate_l{layer_id}"] = record["gate"]
            results[f"titan_inject_delta_l{layer_id}"] = record["delta"]
            results[f"titan_weighted_rms_ratio_l{layer_id}"] = record[
                "weighted_rms_ratio"
            ]
            results[f"titan_region_memory_key_l{layer_id}"] = record[
                "memory_keys"
            ]
            results[f"titan_region_memory_read_l{layer_id}"] = record[
                "memory_values"
            ]
        dynamic = callback_records.get("dynamic_writeback")
        if dynamic is not None:
            results.update(
                {
                    "titan_dynamic_write_layer": int(dynamic["layer"]),
                    "titan_dynamic_region_context": dynamic["context"],
                    "titan_dynamic_region_weight": dynamic["region_weight"],
                    "titan_dynamic_key_gate": dynamic["gate"],
                    "titan_dynamic_key_delta": dynamic["delta"],
                    "titan_dynamic_base_keys": dynamic["base_keys"],
                    "titan_dynamic_updated_keys": dynamic["updated_keys"],
                    "titan_dynamic_alpha": dynamic["alpha"],
                }
            )


_STARPATH_OPTIONS = frozenset(['align_cosine_margin', 'align_min_anchor_norm', 'align_min_region_mass', 'align_min_region_occupancy', 'align_min_total_mass', 'atlas_alpha_init', 'atlas_alpha_max', 'atlas_count_scale', 'atlas_temperature', 'coarse_alpha_init', 'coarse_alpha_max', 'coarse_compatibility_dim', 'coarse_context_dim', 'coarse_pathway_dim', 'coarse_pathway_dropout', 'coarse_uot_epsilon', 'coarse_uot_iterations', 'coarse_uot_tau_source', 'coarse_uot_tau_target', 'dynamic_alpha_init', 'dynamic_alpha_max', 'inject_alpha_init', 'inject_alpha_max', 'inject_min_confidence_sum', 'inject_rms_epsilon', 'loss_w_align', 'loss_w_region', 'min_pathway_genes', 'morphology_centroids', 'omic_sizes', 'pathway_dim', 'pathway_dropout', 'pathway_hidden_dim', 'pathway_names', 'pathway_signature_path', 'pathway_type', 'prototype_path', 'region_assignment_dim', 'region_assignment_log_weight', 'region_candidate_topk', 'region_coord_num_freqs', 'region_loss_spatial_weight', 'region_min_spatial_scale', 'region_norm_epsilon', 'region_num', 'region_score_spatial_weight', 'region_seed_knn', 'region_temperature', 'rna_gene_seq', 'st_gene_seq', 'transport_epsilon', 'transport_mass_epsilon', 'transport_min_reference_mass', 'transport_occupancy_log_weight', 'transport_reliability_beta', 'transport_sinkhorn_iters', 'transport_tau_source', 'transport_tau_target'])

# Public joint coarse-to-fine model.
class STARPath(_FineSTARPath):
    """Coarse-to-fine survival model with one private TITAN encoder.

    Injection and trainable layers are independent, zero-indexed block lists.
    Memory keys are written once after the block immediately before the final
    injection. The default [2, 4] therefore writes after block 3; [2] writes
    after block 1, and [1, 2, 4] still writes after block 3. A lone injection at
    block 0 is invalid because it has no preceding block for contextual memory.
    """

    def __init__(
        self,
        input_dim: int = 768,
        n_classes: int = 4,
        mode: str = "classification",
        ds_num: Optional[int] = None,
        *,
        titan_inject_layers: Sequence[int] = (2, 4),
        titan_trainable_layers: Sequence[int] = (2, 3, 4, 5),
        model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        unknown = set(kwargs).difference(_STARPATH_OPTIONS)
        if unknown:
            raise TypeError(f"Unknown STARPath options: {sorted(unknown)}")
        kwargs["titan_inject_layers"] = titan_inject_layers
        kwargs["titan_trainable_layers"] = titan_trainable_layers
        kwargs["model_path"] = model_path

        omic_sizes = kwargs.get("omic_sizes")
        if omic_sizes is None:
            raise ValueError(
                "STARPath requires omic_sizes for its 50 coarse "
                "Hallmark SNNs"
            )

        # The shared base owns the sole callback-enabled TITAN, patient
        # classifier, and independent injection/trainability configuration.
        super().__init__(
            input_dim=input_dim,
            n_classes=n_classes,
            mode=mode,
            ds_num=ds_num,
            **kwargs,
        )
        self.coarse_conditioner = CoarseConditioner(
            omic_sizes=omic_sizes,
            prototype_path=kwargs.get("prototype_path"),
            pathway_names=kwargs.get("pathway_names"),
            input_dim=input_dim,
            pathway_dim=int(kwargs.get("coarse_pathway_dim", 256)),
            compatibility_dim=int(
                kwargs.get("coarse_compatibility_dim", 256)
            ),
            context_dim=int(kwargs.get("coarse_context_dim", 256)),
            pathway_dropout=float(
                kwargs.get("coarse_pathway_dropout", 0.25)
            ),
            uot_epsilon=float(kwargs.get("coarse_uot_epsilon", 0.07)),
            uot_tau_source=float(
                kwargs.get("coarse_uot_tau_source", 0.5)
            ),
            uot_tau_target=float(
                kwargs.get("coarse_uot_tau_target", 0.5)
            ),
            uot_iterations=int(
                kwargs.get("coarse_uot_iterations", 50)
            ),
            alpha_init=float(kwargs.get("coarse_alpha_init", 0.03)),
            alpha_max=float(kwargs.get("coarse_alpha_max", 0.10)),
            morphology_centroids=kwargs.get("morphology_centroids"),
        )
        self.coarse_pathway_names = tuple(
            self.coarse_conditioner.pathway_names
        )

    @property
    def coarse(self) -> CoarseConditioner:
        """Stable shorthand without registering the module a second time."""
        return self.coarse_conditioner

    def encode_coarse_rna_pathways(
        self, pathways: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Encode the separately normalized coarse RNA view once per patient."""
        if not isinstance(pathways, (list, tuple)):
            raise TypeError(
                "Coarse RNA must be a sequence of 50 pathway tensors"
            )
        parameter = next(self.coarse_conditioner.parameters())
        moved = []
        for index, values in enumerate(pathways):
            if not torch.is_tensor(values):
                raise TypeError(
                    f"Coarse pathway {index} is not a torch.Tensor"
                )
            moved.append(
                values.to(device=parameter.device, dtype=parameter.dtype)
            )
        return self.coarse_conditioner.encode_rna_pathways(moved)

    @staticmethod
    def _record_coarse_results(
        results: Dict[str, Any], diagnostics: Dict[str, Any]
    ) -> None:
        results.update(
            {
                "coarse_pathway_names": list(
                    diagnostics["pathway_names"]
                ),
                "coarse_active_morphology_ids": diagnostics[
                    "active_morphology_ids"
                ],
                "coarse_morphology_occupancy": diagnostics[
                    "morphology_occupancy"
                ],
                "coarse_sampled_morphology_ids": diagnostics[
                    "sampled_morphology_ids"
                ],
                "coarse_transport": diagnostics["transport"],
                "coarse_transport_cost": diagnostics["transport_cost"],
                "coarse_transport_row_mass": diagnostics[
                    "transport_row_mass"
                ],
                "coarse_transport_col_mass": diagnostics[
                    "transport_col_mass"
                ],
                "coarse_transport_total_mass": diagnostics[
                    "transport_total_mass"
                ],
                "coarse_prototype_molecular_context": diagnostics[
                    "prototype_molecular_context"
                ],
                "coarse_alpha": diagnostics["coarse_alpha"],
                "coarse_residual_rms_ratio": diagnostics[
                    "residual_rms_ratio"
                ],
                "fine_region_patch_stream": "raw",
                "titan_patch_stream": "coarse_conditioned",
            }
        )

    def forward(
        self,
        x1,
        *,
        st,
        rna,
        coords,
        patch_size_lv0,
        coarse_pathway_tokens,
        morphology_tokens,
        morphology_occupancy,
        morphology_valid,
        route_atlas=None,
        route_atlas_counts=None,
        route_atlas_valid=None,
        st_gene_seq=None,
        rna_gene_seq=None,
        results_dict=None,
        slide_id=None,
        return_attn=False,
        return_slide_features_only=False,
    ):
        del slide_id, return_attn
        if results_dict is None:
            results_dict = {}
        region_parameter = next(self.region_constructor.parameters())
        raw_x, coords, st, rna, patch_size = validate_forward_inputs(
            x1=x1,
            coords=coords,
            st=st,
            rna=rna,
            expected_st_gene_seq=self.st_gene_seq,
            expected_rna_gene_seq=self.rna_gene_seq,
            st_gene_seq=st_gene_seq,
            rna_gene_seq=rna_gene_seq,
            input_dim=self.input_dim,
            patch_size_lv0=patch_size_lv0,
            model_device=region_parameter.device,
            model_dtype=region_parameter.dtype,
        )
        if coarse_pathway_tokens is None:
            raise ValueError(
                "STARPath requires coarse_pathway_tokens encoded once per patient "
                "with encode_coarse_rna_pathways"
            )
        for name, values in (
            ("morphology_tokens", morphology_tokens),
            ("morphology_occupancy", morphology_occupancy),
            ("morphology_valid", morphology_valid),
        ):
            if values is None:
                raise ValueError(f"STARPath requires {name}")
        # Fine construction deliberately sees raw_x only.  Nothing produced by
        # the coarse branch can move the morphology boundaries or regional ST.
        region = self.region_constructor(raw_x, coords)
        conditioned_x, coarse_diagnostics = (
            self.coarse_conditioner.condition_patch_features(
                patch_features=raw_x,
                pathway_tokens=coarse_pathway_tokens,
                morphology_tokens=morphology_tokens,
                morphology_occupancy=morphology_occupancy,
                morphology_valid=morphology_valid,
            )
        )
        regional_st = self._regional_st(region, st)
        rna_paths = self.pathway_encoder.encode_rna(rna)
        st_paths = self.pathway_encoder.encode_regional_st(
            regional_st, region.active_mask
        )

        missing_atlas = [
            name
            for name, values in (
                ("route_atlas", route_atlas),
                ("route_atlas_counts", route_atlas_counts),
                ("route_atlas_valid", route_atlas_valid),
            )
            if values is None
        ]
        if missing_atlas:
            raise ValueError(
                f"STARPath requires {', '.join(missing_atlas)}"
            )
        if self.route_atlas is None:
            raise RuntimeError("Route Atlas is enabled but was not initialized")
        atlas_output = self.route_atlas(
            route_atlas=route_atlas,
            route_atlas_counts=route_atlas_counts,
            route_atlas_valid=route_atlas_valid,
            sampled_anchor_ids=coarse_diagnostics[
                "sampled_morphology_ids"
            ],
            region_assignment=region.q_soft,
            active_region_mask=region.active_mask,
        )
        transport = self.transport(
            rna_pathway=rna_paths.tokens,
            st_pathway=st_paths.tokens,
            rna_pathway_valid=rna_paths.valid,
            st_pathway_valid=st_paths.valid,
            st_pathway_reliability=st_paths.reliability,
            region_visual=region.visual,
            occupancy=region.occupancy,
            active_region_mask=region.active_mask,
            route_cost_delta=(
                atlas_output.cost_delta
            ),
        )
        memory = self.memory_builder(region, transport)

        callback_records: Dict[str, Any] = {}
        interaction_state = (
            (RegionInteractionState(keys=memory.keys))
        )
        callback = self.memory_reader.make_callback(
            memory=memory,
            transport=transport,
            value_adapter=self.memory_builder.adapt_values,
            records=callback_records,
            interaction_state=interaction_state,
        )
        callback_context = torch.cat(
            [
                region.q_soft,
                region.candidate_mask.to(region.q_soft.dtype),
            ],
            dim=-1,
        )
        if self.dynamic_memory_writer is None or interaction_state is None:
            raise RuntimeError("STARPath dynamic writer was not initialized")
        write_layer = self.titan_inject_layers[-1] - 1
        post_block_callback = self.dynamic_memory_writer.make_post_callback(
            write_layer=write_layer,
            memory=memory,
            interaction_state=interaction_state,
            records=callback_records,
        )

        # This is the only TITAN call.  Only its patch input is conditioned;
        # coords and the fine callback context retain the raw-patch alignment.
        titan_embedding = self._encode_titan(
            conditioned_x,
            coords,
            patch_size,
            callback,
            callback_context,
            post_block_callback,
        )
        if interaction_state.updates != 1:
            raise RuntimeError(
                "STARPath expected exactly one contextual region-key writeback"
            )
        slide_embedding = self.slide_final_norm(titan_embedding)
        alignment_loss = callback_records.get(
            "alignment_loss", slide_embedding.new_zeros(())
        )
        self._record_results(
            results_dict,
            slide_embedding=slide_embedding,
            region=region,
            transport=transport,
            memory=memory,
            atlas_output=atlas_output,
            alignment_loss=alignment_loss,
            callback_records=callback_records,
        )
        self._record_coarse_results(results_dict, coarse_diagnostics)
        results_dict["aux_loss"] = {
            "terms": {
                "region": {
                    "value": region.loss_region,
                    "weight": self.loss_w_region,
                },
                "alignment": {
                    "value": alignment_loss,
                    "weight": self.loss_w_align,
                },
            },
            "nll_logits": {},
        }
        if return_slide_features_only:
            return None, None, None, results_dict
        logits, y_hat, hazards = self.classify_patient(slide_embedding)
        return logits, y_hat, hazards, results_dict



# Patient-level slide aggregation.
def _validate_slide_batch(
    slide_features: torch.Tensor,
    slide_mask: torch.Tensor,
    input_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate a padded slide batch and return its mask/counts."""
    if not torch.is_tensor(slide_features):
        raise TypeError("slide_features must be a torch.Tensor.")
    if slide_features.ndim != 3:
        raise ValueError(
            "slide_features must have shape [B, S, D], got "
            f"{tuple(slide_features.shape)}."
        )
    if not slide_features.is_floating_point():
        raise TypeError("slide_features must be floating point.")
    if slide_features.shape[0] == 0 or slide_features.shape[1] == 0:
        raise ValueError("slide_features must contain at least one patient and slide slot.")
    if slide_features.shape[2] != input_dim:
        raise ValueError(
            f"Expected slide feature dimension {input_dim}, got "
            f"{slide_features.shape[2]}."
        )

    if not torch.is_tensor(slide_mask):
        raise TypeError("slide_mask must be a torch.Tensor.")
    if slide_mask.dtype is not torch.bool:
        raise TypeError("slide_mask must have dtype torch.bool.")
    if slide_mask.ndim != 2 or slide_mask.shape != slide_features.shape[:2]:
        raise ValueError(
            "slide_mask must have shape [B, S] matching slide_features; got "
            f"{tuple(slide_mask.shape)} for features "
            f"{tuple(slide_features.shape)}."
        )
    if slide_mask.device != slide_features.device:
        raise ValueError("slide_mask and slide_features must be on the same device.")

    valid_counts = slide_mask.sum(dim=1)
    if bool((valid_counts == 0).any().item()):
        raise ValueError("Every patient must have at least one valid slide.")

    valid_positions = slide_mask.unsqueeze(-1)
    invalid_valid_values = (~torch.isfinite(slide_features)) & valid_positions
    if bool(invalid_valid_values.any().item()):
        raise ValueError("Valid slide features must contain only finite values.")

    return slide_mask, valid_counts


def _masked_slide_mean(
    slide_features: torch.Tensor,
    slide_mask: torch.Tensor,
    valid_counts: torch.Tensor,
) -> torch.Tensor:
    """Mean over valid slides while making padded values completely inert."""
    safe_features = torch.where(
        slide_mask.unsqueeze(-1), slide_features, torch.zeros_like(slide_features)
    )
    return safe_features.sum(dim=1) / valid_counts.to(slide_features.dtype).unsqueeze(-1)


def _single_slide_identity(
    aggregated: torch.Tensor,
    slide_features: torch.Tensor,
    slide_mask: torch.Tensor,
    valid_counts: torch.Tensor,
) -> torch.Tensor:
    """Explicitly bypass an aggregator for patients with exactly one slide."""
    first_valid_index = slide_mask.to(torch.int64).argmax(dim=1)
    batch_index = torch.arange(slide_features.shape[0], device=slide_features.device)
    only_slide = slide_features[batch_index, first_valid_index]
    return torch.where((valid_counts == 1).unsqueeze(-1), only_slide, aggregated)


class MeanSlideAggregator(nn.Module):
    """Masked arithmetic mean of slide embeddings."""

    def __init__(self, input_dim: int = 768) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self.input_dim = int(input_dim)

    def forward(
        self, slide_features: torch.Tensor, slide_mask: torch.Tensor
    ) -> torch.Tensor:
        slide_mask, valid_counts = _validate_slide_batch(
            slide_features, slide_mask, self.input_dim
        )
        patient_features = _masked_slide_mean(
            slide_features, slide_mask, valid_counts
        )
        return _single_slide_identity(
            patient_features, slide_features, slide_mask, valid_counts
        )


class GatedSlideAttentionAggregator(nn.Module):
    """Ilse-style tanh/sigmoid gated attention over a patient's slides."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 128) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.attention_tanh = nn.Linear(self.input_dim, self.hidden_dim)
        self.attention_sigmoid = nn.Linear(self.input_dim, self.hidden_dim)
        self.attention_score = nn.Linear(self.hidden_dim, 1)

    def forward(
        self, slide_features: torch.Tensor, slide_mask: torch.Tensor
    ) -> torch.Tensor:
        slide_mask, valid_counts = _validate_slide_batch(
            slide_features, slide_mask, self.input_dim
        )
        safe_features = torch.where(
            slide_mask.unsqueeze(-1), slide_features, torch.zeros_like(slide_features)
        )

        gated = torch.tanh(self.attention_tanh(safe_features)) * torch.sigmoid(
            self.attention_sigmoid(safe_features)
        )
        scores = self.attention_score(gated).squeeze(-1)
        scores = scores.masked_fill(~slide_mask, float("-inf"))
        attention = torch.softmax(scores, dim=1)
        attention = attention.masked_fill(~slide_mask, 0.0)
        patient_features = torch.sum(attention.unsqueeze(-1) * safe_features, dim=1)

        return _single_slide_identity(
            patient_features, slide_features, slide_mask, valid_counts
        )


class _SetAttentionBlock(nn.Module):
    """Multi-head attention followed by a position-wise feed-forward block."""

    def __init__(self, latent_dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(latent_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(latent_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, latent_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys_and_values: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            queries,
            keys_and_values,
            keys_and_values,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        output = self.attention_norm(queries + attended)
        return self.feed_forward_norm(output + self.feed_forward(output))


class SetTransformerSlideAggregator(nn.Module):
    """One SAB plus one-seed PMA with a masked-mean residual connection."""

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
    ) -> None:
        super().__init__()
        if min(input_dim, latent_dim, num_heads, ffn_dim) <= 0:
            raise ValueError("All Set Transformer dimensions must be positive.")
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads.")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)

        self.input_projection = nn.Linear(self.input_dim, self.latent_dim)
        self.self_attention_block = _SetAttentionBlock(
            self.latent_dim, self.num_heads, self.ffn_dim
        )
        self.pooling_attention_block = _SetAttentionBlock(
            self.latent_dim, self.num_heads, self.ffn_dim
        )
        self.pooling_seed = nn.Parameter(torch.empty(1, 1, self.latent_dim))
        self.output_projection = nn.Linear(self.latent_dim, self.input_dim)
        nn.init.normal_(self.pooling_seed, mean=0.0, std=0.02)

    def forward(
        self, slide_features: torch.Tensor, slide_mask: torch.Tensor
    ) -> torch.Tensor:
        slide_mask, valid_counts = _validate_slide_batch(
            slide_features, slide_mask, self.input_dim
        )
        mean_features = _masked_slide_mean(slide_features, slide_mask, valid_counts)
        safe_features = torch.where(
            slide_mask.unsqueeze(-1), slide_features, torch.zeros_like(slide_features)
        )
        key_padding_mask = ~slide_mask

        encoded = self.input_projection(safe_features)
        encoded = self.self_attention_block(
            encoded, encoded, key_padding_mask=key_padding_mask
        )
        encoded = torch.where(
            slide_mask.unsqueeze(-1), encoded, torch.zeros_like(encoded)
        )

        seeds = self.pooling_seed.expand(slide_features.shape[0], -1, -1)
        pooled = self.pooling_attention_block(
            seeds, encoded, key_padding_mask=key_padding_mask
        ).squeeze(1)
        patient_features = mean_features + self.output_projection(pooled)

        return _single_slide_identity(
            patient_features, slide_features, slide_mask, valid_counts
        )


def build_slide_aggregator(name: str, **kwargs: int) -> nn.Module:
    """Build a slide aggregator from a compact, case-insensitive name."""
    normalized_name = str(name).strip().lower().replace("-", "_")
    if normalized_name in {"mean", "masked_mean"}:
        return MeanSlideAggregator(**kwargs)
    if normalized_name in {"gated", "gated_attention", "gated_slide_attention"}:
        return GatedSlideAttentionAggregator(**kwargs)
    if normalized_name in {"set", "set_transformer", "settransformer"}:
        return SetTransformerSlideAggregator(**kwargs)
    raise ValueError(
        f"Unknown slide aggregator {name!r}; expected mean, gated, or set_transformer."
    )


class STARPathPatientAdapter(nn.Module):
    """Encode slides independently, aggregate them, and classify one patient."""

    is_unified_adapter = True

    def __init__(self, backbone: nn.Module, slide_aggregator: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.slide_aggregator = slide_aggregator

    def _normalise_payload(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, Mapping):
            raise TypeError("STARPath expects a structured per-patient image payload")
        required = {
            "slides", "coords", "st", "morphology_tokens", "morphology_occupancy",
            "morphology_valid", "slide_ids", "patch_sizes", "route_atlas",
            "route_atlas_counts", "route_atlas_valid",
        }
        missing = required.difference(data)
        if missing:
            raise KeyError(f"STARPath payload is missing {sorted(missing)}")
        resolved = {key: data[key] for key in required}
        lengths = {key: len(value) for key, value in resolved.items()}
        if len(set(lengths.values())) != 1 or next(iter(lengths.values())) == 0:
            raise ValueError(f"STARPath per-slide payload lengths differ: {lengths}")
        return resolved

    def _validate_route_atlas(
        self,
        route_atlas: torch.Tensor,
        route_atlas_counts: torch.Tensor,
        route_atlas_valid: torch.Tensor,
    ) -> None:
        expected_gene_count = len(self.backbone.st_gene_seq)
        if route_atlas.shape[0] != 16:
            raise ValueError(
                "STARPath route_atlas must contain exactly 16 morphology anchors"
            )
        if tuple(route_atlas_counts.shape) != (16,):
            raise ValueError("STARPath route_atlas_counts must have shape [16]")
        if tuple(route_atlas_valid.shape) != (16,):
            raise ValueError("STARPath route_atlas_valid must have shape [16]")
        if route_atlas.shape[1] != expected_gene_count:
            raise ValueError(
                "STARPath route_atlas gene width differs from the backbone ST axis: "
                f"{route_atlas.shape[1]} vs {expected_gene_count}"
            )
        if not route_atlas.is_floating_point():
            raise TypeError("STARPath route_atlas must be floating point")
        if not bool(torch.isfinite(route_atlas).all().detach().item()):
            raise ValueError("STARPath route_atlas contains NaN/Inf values")
        if route_atlas_counts.dtype is torch.bool:
            raise TypeError("STARPath route_atlas_counts must be numeric, not bool")
        if route_atlas_counts.is_floating_point() and not bool(
            torch.isfinite(route_atlas_counts).all().detach().item()
        ):
            raise ValueError("STARPath route_atlas_counts contains NaN/Inf values")
        if bool((route_atlas_counts < 0).any().detach().item()):
            raise ValueError("STARPath route_atlas_counts must be non-negative")
        if route_atlas_valid.dtype is not torch.bool:
            raise TypeError("STARPath route_atlas_valid must have dtype bool")
        expected_valid = route_atlas_counts > 0
        if not torch.equal(route_atlas_valid, expected_valid):
            raise ValueError(
                "STARPath route_atlas_valid must equal route_atlas_counts > 0"
            )
        if not bool(route_atlas_valid.any().detach().item()):
            raise ValueError("STARPath route atlas contains no occupied anchor")
        if bool((route_atlas[~route_atlas_valid] != 0).any().detach().item()):
            raise ValueError("Empty STARPath route-atlas anchors must be exact zero")

    @staticmethod
    def _unbatch_slide_tensor(value: torch.Tensor, expected_dim: int) -> torch.Tensor:
        if value.dim() == expected_dim + 1 and value.shape[0] == 1:
            value = value.squeeze(0)
        if value.dim() != expected_dim:
            raise ValueError(f"Unexpected STARPath tensor shape {tuple(value.shape)}")
        return value

    @staticmethod
    def _normalise_omics(omics):
        if not isinstance(omics, Mapping):
            raise TypeError(
                "Joint STARPath expects omics={'coarse_pathways', 'fine_rna'}"
            )
        missing = {"coarse_pathways", "fine_rna"}.difference(omics)
        if missing:
            raise KeyError(f"STARPath omics payload is missing {sorted(missing)}")
        coarse_pathways = omics["coarse_pathways"]
        if not isinstance(coarse_pathways, (list, tuple)):
            raise TypeError("coarse_pathways must be a sequence of tensors")
        if len(coarse_pathways) != 50:
            raise ValueError("STARPath requires exactly 50 coarse Hallmark pathways")
        fine_rna = omics["fine_rna"]
        if not torch.is_tensor(fine_rna):
            raise TypeError("fine_rna must be a tensor")
        if fine_rna.dim() == 2 and fine_rna.shape[0] == 1:
            fine_rna = fine_rna.squeeze(0)
        if fine_rna.dim() != 1:
            raise ValueError("fine_rna must have shape [G] or [1,G]")
        return coarse_pathways, fine_rna

    def forward(
        self,
        data,
        omics=None,
        *,
        label=None,
        censorship=None,
        discrete_label=None,
        survival_time=None,
        loss_fn=None,
        return_attn=False,
        batch=None,
        defer_survival_loss=False,
        include_auxiliary_loss=True,
        **_: Any,
    ):
        from .components import process_surv
        from .survival_adapter import UnifiedSurvivalAdapter

        del batch
        payload = self._normalise_payload(data)
        coarse_pathways, fine_rna = self._normalise_omics(omics)

        # The SNNs contain AlphaDropout, so this patient-level representation
        # must be computed once and shared by every slide from the patient.
        coarse_pathway_tokens = self.backbone.encode_coarse_rna_pathways(
            coarse_pathways
        )

        slide_features = []
        aux_values = defaultdict(list)
        aux_weights: Dict[str, float] = {}
        slide_count = len(payload["slides"])
        route_atlas_values = (
            (payload["route_atlas"])
        )
        route_atlas_count_values = (
            (payload["route_atlas_counts"])
        )
        route_atlas_valid_values = (
            (payload["route_atlas_valid"])
        )
        for (
            features,
            coords,
            st,
            morphology_tokens,
            morphology_occupancy,
            morphology_valid,
            slide_id,
            patch_size,
            route_atlas,
            route_atlas_counts,
            route_atlas_valid,
        ) in zip(
            payload["slides"],
            payload["coords"],
            payload["st"],
            payload["morphology_tokens"],
            payload["morphology_occupancy"],
            payload["morphology_valid"],
            payload["slide_ids"],
            payload["patch_sizes"],
            route_atlas_values,
            route_atlas_count_values,
            route_atlas_valid_values,
        ):
            features = self._unbatch_slide_tensor(features, 2)
            coords = self._unbatch_slide_tensor(coords, 2)
            st = self._unbatch_slide_tensor(st, 2)
            morphology_tokens = self._unbatch_slide_tensor(morphology_tokens, 2)
            morphology_occupancy = self._unbatch_slide_tensor(
                morphology_occupancy, 1
            )
            morphology_valid = self._unbatch_slide_tensor(morphology_valid, 1)
            route_atlas = self._unbatch_slide_tensor(route_atlas, 2)
            route_atlas_counts = self._unbatch_slide_tensor(
                route_atlas_counts, 1
            )
            route_atlas_valid = self._unbatch_slide_tensor(
                route_atlas_valid, 1
            )
            self._validate_route_atlas(
                route_atlas, route_atlas_counts, route_atlas_valid
            )
            if torch.is_tensor(patch_size):
                patch_size = int(patch_size.reshape(-1)[0].item())
            if isinstance(slide_id, (tuple, list)) and len(slide_id) == 1:
                slide_id = slide_id[0]
            backbone_kwargs = dict(
                x1=features,
                coords=coords,
                st=st,
                rna=fine_rna,
                coarse_pathway_tokens=coarse_pathway_tokens,
                patch_size_lv0=int(patch_size),
                morphology_tokens=morphology_tokens,
                morphology_occupancy=morphology_occupancy,
                morphology_valid=morphology_valid,
                slide_id=str(slide_id),
                return_attn=return_attn,
                return_slide_features_only=True,
            )
            backbone_kwargs.update({
                "route_atlas": route_atlas,
                "route_atlas_counts": route_atlas_counts,
                "route_atlas_valid": route_atlas_valid,
            })
            _, _, _, slide_result = self.backbone(**backbone_kwargs)
            slide_features.append(slide_result["slide_feat"].reshape(1, -1))
            for name, term in slide_result.get("aux_loss", {}).get(
                "terms", {}
            ).items():
                value = term["value"] if isinstance(term, Mapping) else term
                weight = (
                    term.get("weight", 1.0)
                    if isinstance(term, Mapping)
                    else 1.0
                )
                aux_values[name].append(value.reshape(()))
                aux_weights[name] = float(weight)

        feature_tensor = torch.stack(slide_features, dim=1)  # [1, slides, 768]
        slide_mask = torch.ones(
            feature_tensor.shape[:2], dtype=torch.bool, device=feature_tensor.device
        )
        patient_feature = self.slide_aggregator(feature_tensor, slide_mask)
        logits = self.backbone.classifier(patient_feature)
        model_payload = {
            "slide_feat": feature_tensor,
            "patient_feat": patient_feature,
            "slide_mask": slide_mask,
            "aux_loss": {
                "terms": {
                    name: {
                        "value": torch.stack(values).mean(),
                        "weight": aux_weights[name],
                    }
                    for name, values in aux_values.items()
                },
                "nll_logits": {},
            },
        }
        results, log_dict = process_surv(
            logits,
            label,
            censorship,
            loss_fn,
            survival_time=survival_time,
            defer_survival_loss=defer_survival_loss,
        )
        UnifiedSurvivalAdapter._add_auxiliary_losses(
            results,
            log_dict,
            model_payload,
            loss_fn=loss_fn,
            label=label,
            censorship=censorship,
            discrete_label=discrete_label,
            enabled=(
                self.training
                and loss_fn is not None
                and include_auxiliary_loss
            ),
        )
        results.update({key: value for key, value in model_payload.items() if key != "aux_loss"})
        return results, log_dict




__all__ = ["STARPath", "STARPathPatientAdapter", "build_slide_aggregator", "MeanSlideAggregator", "GatedSlideAttentionAggregator", "SetTransformerSlideAggregator", "load_morphology_centroids", "nearest_morphology_assignments", "summarize_full_slide_morphology", "summarize_full_slide_morphology_route_atlas"]
