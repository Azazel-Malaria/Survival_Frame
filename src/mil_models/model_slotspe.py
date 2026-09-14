"""SlotSPE survival backbone, adapted from the official release.

Upstream: zylvemvet/SlotSPE, commit 02051a2083add7b427727e0200e4263516903561.
RNA is an explicitly aligned vector sliced by official signature intersections.
The shared adapter owns survival losses; the model declares its NLL auxiliary
heads and reconstruction term independently of the primary NLL/Cox head.
"""
import csv
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def _norm_gene_name(gene: str) -> str:
    return str(gene).strip().upper()

def _load_signature_columns(signature_path: str) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """
    Read SlotSPE/SurvPath-style signature CSV files.

    Expected format: each column is one pathway/signature, and each non-empty cell
    under that column is a gene symbol. This matches SlotSPE's dataset_csv/signatures/*.csv.
    """
    if not os.path.exists(signature_path):
        raise FileNotFoundError(f'Signature file not found: {signature_path}')
    with open(signature_path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f'Empty signature file: {signature_path}')
    header = [h.strip() for h in rows[0]]
    keep_cols: List[int] = []
    names: List[str] = []
    for i, h in enumerate(header):
        if h == '' or h.lower().startswith('unnamed'):
            continue
        keep_cols.append(i)
        names.append(h)
    if not keep_cols:
        raise ValueError(f'No valid pathway columns found in {signature_path}')
    columns: List[List[str]] = [[] for _ in keep_cols]
    for row in rows[1:]:
        for out_i, col_i in enumerate(keep_cols):
            if col_i >= len(row):
                continue
            gene = row[col_i].strip()
            if gene and gene.lower() != 'nan':
                columns[out_i].append(gene)
    signatures: List[Tuple[str, Tuple[str, ...]]] = []
    for name, genes in zip(names, columns):
        seen = set()
        uniq: List[str] = []
        for gene in genes:
            g = _norm_gene_name(gene)
            if g and g not in seen:
                seen.add(g)
                uniq.append(g)
        if uniq:
            signatures.append((name, tuple(uniq)))
    if not signatures:
        raise ValueError(f'No non-empty pathways found in {signature_path}')
    return tuple(signatures)

def build_pathway_gene_indices(gene_seq: Sequence[str], signature_path: str, min_genes: int=1) -> Tuple[List[str], List[List[int]]]:
    """Use each official signature's sorted intersection with the RNA gene axis."""
    if min_genes < 1:
        raise ValueError('min_genes must be positive')
    normalized = [_norm_gene_name(gene) for gene in gene_seq]
    if len(set(normalized)) != len(normalized) or any((not gene for gene in normalized)):
        raise ValueError('RNA gene names must be nonempty and unique')
    gene_to_idx = {gene: idx for idx, gene in enumerate(normalized)}
    names, indices = ([], [])
    for name, genes in _load_signature_columns(signature_path):
        idxs = [gene_to_idx[gene] for gene in sorted(set(genes)) if gene in gene_to_idx]
        if len(idxs) >= min_genes:
            names.append(name)
            indices.append(idxs)
    if not indices:
        raise ValueError('No signature intersects the configured RNA gene axis')
    return (names, indices)

def SNN_Block(dim1: int, dim2: int, dropout: float=0.25) -> nn.Sequential:
    """SlotSPE omics block: Linear + ELU + AlphaDropout."""
    return nn.Sequential(nn.Linear(dim1, dim2), nn.ELU(), nn.AlphaDropout(p=dropout, inplace=False))

def WSI_Mlp(dim_in: int, feat_dim: int) -> nn.Sequential:
    hidden_dim = dim_in
    return nn.Sequential(nn.Linear(dim_in, hidden_dim), nn.ReLU(inplace=False), nn.Linear(hidden_dim, feat_dim))

class MultiHeadSlotAttention(nn.Module):
    """Multi-head Slot Attention following SlotSPE's official implementation."""

    def __init__(self, num_slots: int, dim: int, heads: int=8, dim_head: int=64, iters: int=3, eps: float=1e-08, hidden_dim: int=128):
        super().__init__()
        if int(dim_head) <= 0:
            raise ValueError(f'dim_head must be positive, got {dim_head}')
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.heads = heads
        self.dim_head = int(dim_head)
        self.scale = dim ** (-0.5)
        dim_inner = self.dim_head * heads
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim_inner)
        self.to_k = nn.Linear(dim, dim_inner)
        self.to_v = nn.Linear(dim, dim_inner)
        self.combine_heads = nn.Linear(dim_inner, dim)
        self.gru = nn.GRUCell(dim, dim)
        hidden_dim = max(dim, hidden_dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.dim_head).transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, n, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * d)

    def forward(self, inputs: torch.Tensor, num_slots: Optional[int]=None) -> torch.Tensor:
        b, _, _ = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn_like(mu)
        inputs = self.norm_input(inputs)
        k = self._split_heads(self.to_k(inputs))
        v = self._split_heads(self.to_v(inputs))
        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self._split_heads(self.to_q(slots_norm))
            dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = dots.softmax(dim=-2)
            attn = F.normalize(attn + self.eps, p=1, dim=-1)
            updates = torch.matmul(attn, v)
            updates = self.combine_heads(self._merge_heads(updates))
            slots = self.gru(updates.reshape(-1, self.dim), slots_prev.reshape(-1, self.dim)).view(b, n_s, self.dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots

def _gumbel_noise(t: torch.Tensor, eps: float=1e-20) -> torch.Tensor:
    noise = torch.rand_like(t).clamp(min=eps, max=1.0 - eps)
    return -torch.log((-torch.log(noise)).clamp(min=eps))

def _parallel_topk_st(logits: torch.Tensor, k: int=1, temperature: float=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    noised_logits = logits + _gumbel_noise(logits)
    topk_indices = noised_logits.topk(k=k, dim=-1).indices
    hard_k_hot = torch.zeros_like(logits)
    hard_k_hot.scatter_(1, topk_indices, 1.0)
    soft_k_hot = k * F.softmax(noised_logits / temperature, dim=-1)
    y = hard_k_hot + soft_k_hot - soft_k_hot.detach()
    return (y, topk_indices)

def _relaxed_topk(logits: torch.Tensor, k: int, temperature: float=1.0) -> torch.Tensor:
    scores = logits
    soft_k_hot = torch.zeros_like(logits)
    for _ in range(k):
        probs = F.softmax(scores / temperature, dim=-1)
        soft_k_hot = soft_k_hot + probs
        scores = scores + torch.log((1.0 - probs).clamp(min=1e-20))
    return soft_k_hot

def _gumbel_topk_st(logits: torch.Tensor, k: int=1, temperature: float=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    noised_logits = logits + _gumbel_noise(logits)
    topk_indices = noised_logits.topk(k=k, dim=-1).indices
    hard_k_hot = torch.zeros_like(logits)
    hard_k_hot.scatter_(1, topk_indices, 1.0)
    soft_k_hot = _relaxed_topk(noised_logits, k=k, temperature=temperature)
    y = hard_k_hot + soft_k_hot - soft_k_hot.detach()
    return (y, topk_indices)

class MoESlotDecoder(nn.Module):
    """SlotSPE sparse slot-level decoder with straight-through top-k selection."""

    def __init__(self, dim: int, num_slots: int, num_classes: int=2, temperature: float=0.01, topk_ratio: float=0.25, top_k_method: str='parallel_topk_st'):
        super().__init__()
        self.num_slots = num_slots
        self.temperature = temperature
        self.top_k_method = top_k_method
        self.k = max(1, int(num_slots * topk_ratio))
        self.map = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.decoder = nn.Linear(dim, num_classes)
        self.pred_keep_slot = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = self.map(slots)
        slot_logits = self.decoder(slots)
        keep_score = self.pred_keep_slot(slots).squeeze(-1)
        if self.top_k_method == 'gumbel_topk_st':
            hard_keep, _ = _gumbel_topk_st(keep_score, temperature=self.temperature, k=self.k)
        elif self.top_k_method == 'parallel_topk_st':
            hard_keep, _ = _parallel_topk_st(keep_score, temperature=self.temperature, k=self.k)
        else:
            raise ValueError(f'Invalid top_k_method: {self.top_k_method}')
        slot_gate = torch.softmax(keep_score / self.temperature, dim=-1)
        slot_gate = slot_gate * hard_keep
        slot_gate = slot_gate / (slot_gate.sum(dim=1, keepdim=True) + 1e-08)
        logits = torch.einsum('bs,bsc->bc', slot_gate, slot_logits)
        return (logits, slot_gate, hard_keep)

class Mlp(nn.Module):

    def __init__(self, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, drop: float=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * random_tensor.floor()

class DropPath(nn.Module):
    """Per-sample stochastic depth used by the official SlotSPE transformer."""

    def __init__(self, drop_prob: float=0.1):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)

class Attention(nn.Module):

    def __init__(self, dim: int, num_heads: int=8, attn_drop: float=0.1, proj_drop: float=0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        query_mask = None
        if mask is not None:
            mask = mask.bool()
            if mask.shape != (b, n):
                raise ValueError(f'attention mask should have shape {(b, n)}, got {tuple(mask.shape)}')
            if not (mask.sum(dim=1) > 0).all():
                raise ValueError('each sample must keep at least one slot')
            mask_q = mask.unsqueeze(1).unsqueeze(-1)
            mask_k = mask.unsqueeze(1).unsqueeze(2)
            final_mask = mask_q & mask_k
            attn = attn.masked_fill(~final_mask, -1000000000.0)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, c)
        x = self.proj_drop(self.proj(x))
        if query_mask is not None:
            x = x * query_mask
        return x

class Transformer(nn.Module):

    def __init__(self, dim: int, num_heads: int=8, mlp_ratio: float=1.0, drop: float=0.1, drop_path: float=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        mask_f = None
        if mask is not None:
            mask = mask.bool()
            if mask.shape != x.shape[:2]:
                raise ValueError(f'transformer mask should have shape {tuple(x.shape[:2])}, got {tuple(mask.shape)}')
            if not (mask.sum(dim=1) > 0).all():
                raise ValueError('each sample must keep at least one slot')
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        if mask_f is not None:
            x = x * mask_f
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if mask_f is not None:
            x = x * mask_f
        return x

class IterativeCrossAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int=8, iters: int=3, attn_drop: float=0.1, proj_drop: float=0.1, static_kv: bool=True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.iters = iters
        self.static_kv = bool(static_kv)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gru1 = nn.GRUCell(dim, dim)
        self.gru2 = nn.GRUCell(dim, dim)
        self.norm1a = nn.LayerNorm(dim)
        self.norm1b = nn.LayerNorm(dim)
        self.norm2a = nn.LayerNorm(dim)
        self.norm2b = nn.LayerNorm(dim)
        self.mlp1 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.mlp2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, n, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * d)

    def _apply_cross_mask(self, attn: torch.Tensor, q_mask: Optional[torch.Tensor], k_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if q_mask is None or k_mask is None:
            return attn
        q_mask = q_mask.bool()
        k_mask = k_mask.bool()
        if not (q_mask.sum(dim=1) > 0).all():
            raise ValueError('Every cross-attention sample must retain a query slot')
        if not (k_mask.sum(dim=1) > 0).all():
            raise ValueError('Every cross-attention sample must retain a key slot')
        mask = q_mask.unsqueeze(1).unsqueeze(-1) & k_mask.unsqueeze(1).unsqueeze(2)
        return attn.masked_fill(~mask, -1000000000.0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, mask1: Optional[torch.Tensor]=None, mask2: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.static_kv:
            x1_static = x1
            x2_static = x2
            k1 = self._split(self.to_k(x1_static))
            v1 = self._split(self.to_v(x1_static))
            k2 = self._split(self.to_k(x2_static))
            v2 = self._split(self.to_v(x2_static))
        for _ in range(self.iters):
            x1_prev, x2_prev = (x1, x2)
            n1, n2 = (x1.shape[1], x2.shape[1])
            x1n, x2n = (self.norm1a(x1), self.norm2a(x2))
            q1, q2 = (self._split(self.to_q(x1n)), self._split(self.to_q(x2n)))
            if not self.static_kv:
                k1, v1 = (self._split(self.to_k(x1n)), self._split(self.to_v(x1n)))
                k2, v2 = (self._split(self.to_k(x2n)), self._split(self.to_v(x2n)))
            attn1 = torch.matmul(q1, k2.transpose(-2, -1)) * self.scale
            attn2 = torch.matmul(q2, k1.transpose(-2, -1)) * self.scale
            attn1 = self._apply_cross_mask(attn1, mask1, mask2)
            attn2 = self._apply_cross_mask(attn2, mask2, mask1)
            attn1 = self.attn_drop(attn1.softmax(dim=-1))
            attn2 = self.attn_drop(attn2.softmax(dim=-1))
            update1 = self.proj_drop(self.proj(self._merge(torch.matmul(attn1, v2))))
            update2 = self.proj_drop(self.proj(self._merge(torch.matmul(attn2, v1))))
            x1 = self.gru1(update1.reshape(-1, self.dim), x1_prev.reshape(-1, self.dim)).view(-1, n1, self.dim)
            x2 = self.gru2(update2.reshape(-1, self.dim), x2_prev.reshape(-1, self.dim)).view(-1, n2, self.dim)
            x1 = x1 + self.mlp1(self.norm1b(x1))
            x2 = x2 + self.mlp2(self.norm2b(x2))
        return (x1, x2)

class IterativeCrossAttTransformer(nn.Module):

    def __init__(self, dim: int, num_heads: int=8, iters: int=3, drop: float=0.1, drop_path: float=0.1, static_kv: bool=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = IterativeCrossAttention(dim, num_heads=num_heads, iters=iters, attn_drop=drop, proj_drop=drop, static_kv=static_kv)
        self.mlp = Mlp(dim, hidden_features=dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, mask1: Optional[torch.Tensor]=None, mask2: Optional[torch.Tensor]=None) -> torch.Tensor:
        y1, y2 = self.attn(self.norm1(x1), self.norm1(x2), mask1=mask1, mask2=mask2)
        x1 = x1 + self.drop_path(y1)
        x2 = x2 + self.drop_path(y2)
        x1 = x1 + self.drop_path(self.mlp(self.norm2(x1)))
        x2 = x2 + self.drop_path(self.mlp(self.norm2(x2)))
        return torch.cat([x1, x2], dim=1)

class ReconstructionHead(nn.Module):
    """Auxiliary reconstruction head used by SlotSPE."""

    def __init__(self, dim: int, num_heads: int=8, mode: str='range_init', num_queries: Optional[int]=None):
        super().__init__()
        self.mode = mode
        self.dim = dim
        if mode == 'range_init':
            if num_queries is None:
                raise ValueError('num_queries must be provided for range_init')
            self.query_embedding = nn.Embedding(num_queries, dim)
            self.register_buffer('query_ids', torch.arange(num_queries, dtype=torch.long))
        elif mode == 'wsi_patches':
            self.frozen_mlp = nn.Linear(dim, dim)
            for p in self.frozen_mlp.parameters():
                p.requires_grad = False
        else:
            self.token = nn.Parameter(torch.randn(1, 1, dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def _make_query(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        if self.mode == 'range_init':
            return self.query_embedding(self.query_ids).unsqueeze(0).expand(b, -1, -1)
        if self.mode == 'wsi_patches':
            with torch.no_grad():
                return self.frozen_mlp(x.detach())
        return self.token.expand(b, n, -1)

    def forward(self, x: torch.Tensor, slots: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        query = self._make_query(x)
        key_padding_mask = None
        if mask is not None:
            mask = mask.bool()
            if not (mask.sum(dim=1) > 0).all():
                raise ValueError('Every reconstruction sample must retain a slot')
            key_padding_mask = ~mask
        recon, _ = self.cross_attn(query, slots, slots, key_padding_mask=key_padding_mask)
        return recon + self.mlp(self.norm(recon))

class SlotSPE(nn.Module):
    """Official SlotSPE computation with explicit RNA and survival interfaces."""

    def __init__(
        self, input_dim=768, n_classes=4, mode="survival", *,
        signature_path, rna_gene_seq, aux_nll_bins=4, pathway_type="combine",
        wsi_projection_dim=256, num_heads=8, slot_dim_head=64,
        slot_num_wsi=8, slot_num_omics=8, slot_iters=10, cross_iters=3,
        temperature=0.01, topk_ratio=0.25, top_k_method="parallel_topk_st",
        static_kv=True, lambda_recon_loss=0.01, lambda_wsi_aux_nll=1.0,
        lambda_omics_aux_nll=1.0, use_aux_loss=True, min_pathway_genes=1,
        expected_num_pathways=None,
    ):
        super().__init__()
        if mode != "survival":
            raise ValueError("SlotSPE is configured for survival prediction")
        self.mode = mode
        self.n_classes = int(n_classes)
        self.aux_nll_bins = int(aux_nll_bins)
        if self.n_classes < 1 or self.aux_nll_bins < 2:
            raise ValueError('Survival output must be positive and auxiliary NLL needs at least two bins')
        self.implementation_protocol = 'official_release_math'
        self.pathway_type = str(pathway_type).strip().lower()
        if self.pathway_type not in {'hallmarks', 'xena', 'combine'}:
            raise ValueError('Unsupported pathway_type={!r}; expected hallmarks, xena, or combine.'.format(self.pathway_type))
        self.signature_path = signature_path
        self.min_pathway_genes = int(min_pathway_genes)
        self.static_kv = bool(static_kv)
        self.wsi_embedding_dim = int(input_dim)
        self.projection_dim = int(wsi_projection_dim)
        self.num_heads = int(num_heads)
        self.slot_dim_head = int(slot_dim_head)
        if self.slot_dim_head <= 0:
            raise ValueError('slot_dim_head must be positive')
        self.slot_num_wsi = int(slot_num_wsi)
        self.slot_num_omics = int(slot_num_omics)
        self.slot_iters = int(slot_iters)
        self.cross_iters = int(cross_iters)
        self.temperature = float(temperature)
        self.topk_ratio = float(topk_ratio)
        self.top_k_method = top_k_method
        self.lambda_recon_loss = float(lambda_recon_loss)
        self.lambda_wsi_aux_nll = float(lambda_wsi_aux_nll)
        self.lambda_omics_aux_nll = float(lambda_omics_aux_nll)
        aux_weights = (self.lambda_recon_loss, self.lambda_wsi_aux_nll, self.lambda_omics_aux_nll)
        if any((not math.isfinite(weight) or weight < 0.0 for weight in aux_weights)):
            raise ValueError('SlotSPE auxiliary-loss weights must be finite and non-negative.')
        self.use_aux_loss = bool(use_aux_loss)
        if rna_gene_seq is None:
            raise ValueError('SlotSPE requires rna_gene_seq from the configured bulk RNA table.')
        gene_seq = tuple((str(gene).strip() for gene in rna_gene_seq))
        if not gene_seq or any((not gene for gene in gene_seq)):
            raise ValueError('SlotSPE received an empty or invalid rna_gene_seq.')
        pathway_names, pathway_indices = build_pathway_gene_indices(gene_seq=gene_seq, signature_path=self.signature_path, min_genes=self.min_pathway_genes)
        if not pathway_indices:
            raise RuntimeError('No pathway could be built from gene_seq and signature file.')
        self.gene_seq = tuple(gene_seq)
        self.pathway_names = pathway_names
        self.pathway_indices = [torch.as_tensor(idxs, dtype=torch.long) for idxs in pathway_indices]
        self.omic_sizes = [len(idxs) for idxs in pathway_indices]
        self.num_pathways = len(self.pathway_indices)
        if expected_num_pathways is not None and self.num_pathways != int(expected_num_pathways):
            raise ValueError(f'SlotSPE pathway construction produced {self.num_pathways} non-empty pathways; expected {int(expected_num_pathways)} for the configured RNA source')
        self.wsi_mlp = WSI_Mlp(dim_in=self.wsi_embedding_dim, feat_dim=self.projection_dim)
        self.sig_networks = nn.ModuleList([nn.Sequential(SNN_Block(dim1=size, dim2=self.projection_dim), SNN_Block(dim1=self.projection_dim, dim2=self.projection_dim, dropout=0.25)) for size in self.omic_sizes])
        self.slot_attention_wsi = MultiHeadSlotAttention(dim=self.projection_dim, num_slots=self.slot_num_wsi, iters=self.slot_iters, heads=self.num_heads, dim_head=self.slot_dim_head)
        self.slot_attention_omic = MultiHeadSlotAttention(dim=self.projection_dim, num_slots=self.slot_num_omics, iters=self.slot_iters, heads=self.num_heads, dim_head=self.slot_dim_head)
        self.slot_decoder_wsi = MoESlotDecoder(dim=self.projection_dim, num_slots=self.slot_num_wsi, num_classes=self.aux_nll_bins, temperature=self.temperature, topk_ratio=self.topk_ratio, top_k_method=self.top_k_method)
        self.slot_decoder_omic = MoESlotDecoder(dim=self.projection_dim, num_slots=self.slot_num_omics, num_classes=self.aux_nll_bins, temperature=self.temperature, topk_ratio=self.topk_ratio, top_k_method=self.top_k_method)
        self.self_attention_wsi = Transformer(dim=self.projection_dim, num_heads=self.num_heads)
        self.self_attention_omic = Transformer(dim=self.projection_dim, num_heads=self.num_heads)
        self.cross_attention = IterativeCrossAttTransformer(dim=self.projection_dim, num_heads=self.num_heads, iters=self.cross_iters, static_kv=self.static_kv)
        self.to_logits = nn.Linear(self.projection_dim * 3, n_classes)
        self.reconstruction_head_omic = ReconstructionHead(self.projection_dim, num_heads=self.num_heads, mode='range_init', num_queries=self.num_pathways)
        self.reconstruction_head_wsi = ReconstructionHead(self.projection_dim, num_heads=self.num_heads, mode='wsi_patches')
        self.reconstruction_head_joint = ReconstructionHead(self.projection_dim, num_heads=self.num_heads, mode='range_init', num_queries=self.num_pathways)

    def _prepare_wsi(self, x1: torch.Tensor) -> torch.Tensor:
        if x1.dim() == 2:
            x_wsi = x1.unsqueeze(0)
        elif x1.dim() == 3:
            x_wsi = x1
        else:
            raise ValueError(f'x1 should have shape [N,D] or [B,N,D], got {tuple(x1.shape)}')
        if x_wsi.shape[-1] != self.wsi_embedding_dim:
            raise ValueError(f'x1 feature dim mismatch: model was initialized with input_dim={self.wsi_embedding_dim}, but got x1.shape[-1]={x_wsi.shape[-1]}')
        return x_wsi.float()

    def _prepare_rna(self, rna, device, batch_size):
        if rna is None:
            if self.training:
                raise ValueError('SlotSPE training requires RNA; reconstruction is evaluation-only')
            return (torch.zeros(batch_size, len(self.gene_seq), device=device), True)
        rna = torch.as_tensor(rna, device=device, dtype=torch.float32)
        if rna.ndim == 1:
            rna = rna.unsqueeze(0)
        if tuple(rna.shape) != (batch_size, len(self.gene_seq)):
            raise ValueError('SlotSPE RNA must match the patient batch and configured gene axis')
        if not bool(torch.isfinite(rna).all()):
            raise ValueError('SlotSPE RNA contains NaN or Inf')
        return (rna, False)

    def _encode_pathways(self, rna):
        values = [encoder(rna.index_select(1, idx.to(rna.device))) for idx, encoder in zip(self.pathway_indices, self.sig_networks)]
        tokens = torch.stack(values, dim=1)
        mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        return (tokens, mask)

    def _compute_reconstruction_loss(self, x_wsi_proj: torch.Tensor, x_omics: torch.Tensor, slots_wsi: torch.Tensor, slots_omic: torch.Tensor, wsi_keep_slots: torch.Tensor, omic_keep_slots: torch.Tensor) -> torch.Tensor:
        recon_omic = self.reconstruction_head_omic(x_omics, slots_omic, mask=omic_keep_slots.bool())
        recon_mse = F.mse_loss(recon_omic, x_omics.detach())
        slots_omic_from_wsi = self.slot_attention_omic(x_wsi_proj)
        recon_omic_joint = self.reconstruction_head_joint(x_omics, slots_omic_from_wsi)
        recon_mse = recon_mse + F.mse_loss(recon_omic_joint, x_omics.detach())
        recon_wsi = self.reconstruction_head_wsi(x_wsi_proj, slots_wsi, mask=wsi_keep_slots.bool())
        recon_wsi = F.normalize(recon_wsi, dim=-1)
        target_wsi_source = x_wsi_proj
        target_wsi = F.normalize(target_wsi_source, dim=-1)
        recon_loss_wsi = 1.0 - F.cosine_similarity(recon_wsi, target_wsi, dim=-1).mean()
        return recon_mse + recon_loss_wsi

    def forward(self, x1, rna=None, *, return_attn=False, results_dict=None):
        if results_dict is None:
            results_dict = {}
        x_wsi = self._prepare_wsi(x1)
        batch_size = x_wsi.shape[0]
        device = x_wsi.device
        rna_tensor, omic_missing = self._prepare_rna(rna, device=device, batch_size=batch_size)
        x_wsi_proj = self.wsi_mlp(x_wsi)
        x_omics, pathway_mask = self._encode_pathways(rna_tensor)
        if not self.training and omic_missing:
            slots_omic_from_wsi = self.slot_attention_omic(x_wsi_proj)
            x_omics = self.reconstruction_head_joint(x_omics, slots_omic_from_wsi)
            pathway_mask = torch.ones(batch_size, self.num_pathways, device=device, dtype=torch.bool)
        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)
        logits_wsi, wsi_slot_gate, wsi_keep_slots = self.slot_decoder_wsi(slots_wsi)
        logits_omic, omic_slot_gate, omic_keep_slots = self.slot_decoder_omic(slots_omic)
        x_inter = self.cross_attention(slots_wsi, slots_omic)
        wsi_keep_mask = wsi_keep_slots.bool()
        omic_keep_mask = omic_keep_slots.bool()
        wsi_intra = self.self_attention_wsi(slots_wsi, mask=wsi_keep_mask)
        omic_intra = self.self_attention_omic(slots_omic, mask=omic_keep_mask)
        wsi_summary = wsi_intra.mean(dim=1)
        omic_summary = omic_intra.mean(dim=1)
        fused_feature = torch.cat([x_inter.mean(dim=1), wsi_summary, omic_summary], dim=1)
        logits = self.to_logits(fused_feature)
        results_dict['slide_feat'] = fused_feature
        results_dict['pathway_mask'] = pathway_mask
        results_dict['wsi_slot_gate'] = wsi_slot_gate
        results_dict['omic_slot_gate'] = omic_slot_gate
        results_dict['logits_wsi'] = logits_wsi
        results_dict['logits_omic'] = logits_omic
        results_dict['num_pathways'] = self.num_pathways
        results_dict['pathway_names'] = self.pathway_names
        results_dict['implementation_protocol'] = self.implementation_protocol
        aux_loss = {'terms': {}, 'nll_logits': {}}
        if self.use_aux_loss and str(self.mode).lower() in {'survival', 'surv'}:
            aux_loss['nll_logits'] = {'wsi': {'logits': logits_wsi, 'weight': self.lambda_wsi_aux_nll}, 'omics': {'logits': logits_omic, 'weight': self.lambda_omics_aux_nll}}
        if self.training and self.use_aux_loss:
            reconstruction_loss = self._compute_reconstruction_loss(x_wsi_proj=x_wsi_proj, x_omics=x_omics, slots_wsi=slots_wsi, slots_omic=slots_omic, wsi_keep_slots=wsi_keep_slots, omic_keep_slots=omic_keep_slots)
            aux_loss['terms']['reconstruction'] = {'value': reconstruction_loss, 'weight': self.lambda_recon_loss}
        results_dict['aux_loss'] = aux_loss
        if return_attn:
            results_dict['wsi_slots'] = slots_wsi
            results_dict['omic_slots'] = slots_omic
        results_dict['logits'] = logits
        return (logits, None, None, results_dict)
