#!/usr/bin/env python3
"""
Dual-stream Block Diffusion model.

Input layout:
  X_input = cat([X_clean, X_target], dim=1)
  clean length  = n * B
  target length = n * B * K

Rules implemented in this file:
- clean stream tokens are mean-pooled from K PQ IDs
- target stream tokens are noised discrete IDs (with MASK_ID)
- four positional components:
  1) item-level RoPE
  2) layer embedding (clean + target)
  3) slice embedding (target only)
  4) clean type embedding (clean only)
- asymmetric 4-quadrant attention mask
- loss is computed only on masked target tokens
- generation with clean KV cache and layer-by-layer denoising
"""

from __future__ import annotations

import math
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend as _SDPBackend
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel_ctx
except Exception:  # pragma: no cover - backward compatibility for older PyTorch
    _SDPBackend = None
    _sdpa_kernel_ctx = None

_SDPA_BACKEND_CHOICES = {"auto", "math", "flash", "mem_efficient"}


def _normalize_sdp_backend(name: str) -> str:
    out = str(name).strip().lower()
    if out not in _SDPA_BACKEND_CHOICES:
        raise ValueError(
            f"Unsupported sdp backend={name}; expected one of: {sorted(_SDPA_BACKEND_CHOICES)}"
        )
    return out


def _resolve_eff_backend():
    if _SDPBackend is None:
        return None
    if hasattr(_SDPBackend, "EFFICIENT_ATTENTION"):
        return _SDPBackend.EFFICIENT_ATTENTION
    if hasattr(_SDPBackend, "MEM_EFFICIENT_ATTENTION"):
        return _SDPBackend.MEM_EFFICIENT_ATTENTION
    return None


def _sdp_context(backend: str):
    backend = _normalize_sdp_backend(backend)
    if backend == "auto":
        return nullcontext()
    if _sdpa_kernel_ctx is not None and _SDPBackend is not None:
        if backend == "math":
            return _sdpa_kernel_ctx(backends=[_SDPBackend.MATH])
        if backend == "flash":
            flash_backend = getattr(_SDPBackend, "FLASH_ATTENTION", None)
            if flash_backend is None:
                raise RuntimeError("Current torch.nn.attention.SDPBackend has no flash-attention enum")
            return _sdpa_kernel_ctx(backends=[flash_backend])
        eff_backend = _resolve_eff_backend()
        if eff_backend is None:
            raise RuntimeError("Current torch.nn.attention.SDPBackend has no efficient-attention enum")
        return _sdpa_kernel_ctx(backends=[eff_backend])
    # Backward compatibility path for older PyTorch.
    if backend == "math":
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    if backend == "flash":
        return torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
    return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(start_dim=-2)


@dataclass
class CleanKVCache:
    k_per_layer: List[torch.Tensor]
    v_per_layer: List[torch.Tensor]
    clean_item_idx: torch.Tensor  # [L_clean]
    clean_layer_idx: torch.Tensor  # [L_clean]


class BlockDiffusionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
        rope_cache_max_pos: int = 4096,
        train_sdp_backend: str = "auto",
        eval_sdp_backend: str = "math",
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.ffn_dim = int(ffn_dim) if ffn_dim is not None else 4 * self.hidden_dim
        if self.ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be > 0, got {self.ffn_dim}")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE")

        self.rope_base = float(rope_base)
        self.rope_cache_max_pos = int(rope_cache_max_pos)
        self._rope_cache: Dict[Tuple[str, int, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.train_sdp_backend = _normalize_sdp_backend(train_sdp_backend)
        self.eval_sdp_backend = _normalize_sdp_backend(eval_sdp_backend)
        self._sdpa_warned_fallback = False

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_dropout = float(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, self.hidden_dim),
            nn.Dropout(dropout),
        )

    def _rope_tables(self, max_pos: int, device: torch.device, dtype: torch.dtype):
        key = (str(device), int(max_pos), dtype)
        if key in self._rope_cache:
            return self._rope_cache[key]

        inv_freq = 1.0 / (
            self.rope_base
            ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        pos = torch.arange(max_pos, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)  # [max_pos, head_dim/2]
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1).to(dtype=dtype)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1).to(dtype=dtype)
        # Do not cache inference tensors, otherwise a later training pass may
        # hit "Inference tensors cannot be saved for backward".
        if not torch.is_inference_mode_enabled():
            self._rope_cache[key] = (cos, sin)
        return cos, sin

    def _apply_rope(self, x: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
        # x: [B,H,L,Dh], pos_ids: [L] or [B,L]
        # Use a fixed precomputed table to avoid frequent GPU->CPU sync from .item().
        cos, sin = self._rope_tables(max_pos=self.rope_cache_max_pos, device=x.device, dtype=x.dtype)
        pos_ids = pos_ids.long()
        if pos_ids.dim() == 1:
            cos_pos = cos[pos_ids].unsqueeze(0).unsqueeze(0)
            sin_pos = sin[pos_ids].unsqueeze(0).unsqueeze(0)
        elif pos_ids.dim() == 2:
            if pos_ids.size(0) != x.size(0) or pos_ids.size(1) != x.size(2):
                raise ValueError(
                    f"batched pos_ids shape {tuple(pos_ids.shape)} must match x batch/len {(x.size(0), x.size(2))}"
                )
            cos_pos = cos[pos_ids].unsqueeze(1)
            sin_pos = sin[pos_ids].unsqueeze(1)
        else:
            raise ValueError(f"pos_ids must be [L] or [B,L], got {tuple(pos_ids.shape)}")
        return x * cos_pos + _rotate_half(x) * sin_pos

    def _project_qkv(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x_norm.shape
        q = self.q_proj(x_norm).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # attn_mask: [B,Q,K] bool, True=visible. A None mask enables the
        # fused SDPA kernels for fully-visible generation attention.
        attn_bool = None if attn_mask is None else attn_mask.unsqueeze(1)
        dropout_p = self.attn_dropout if self.training else 0.0
        if not q.is_cuda:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bool, dropout_p=dropout_p)

        backend = self.train_sdp_backend if self.training else self.eval_sdp_backend
        if backend == "auto":
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bool, dropout_p=dropout_p)
        try:
            with _sdp_context(backend):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bool, dropout_p=dropout_p)
        except RuntimeError as err:
            # Keep robustness for edge sequence/mask patterns. Fallback does not change model logic.
            if backend != "math":
                if not self._sdpa_warned_fallback:
                    warnings.warn(
                        f"SDPA backend '{backend}' failed once, fallback to 'math'. error={err}",
                        RuntimeWarning,
                    )
                    self._sdpa_warned_fallback = True
                with _sdp_context("math"):
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bool, dropout_p=dropout_p)
            raise

    def forward_full(self, x: torch.Tensor, pos_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: [B,L,D], attn_mask: [B,L,L]
        h = self.norm1(x)
        q, k, v = self._project_qkv(h)
        q = self._apply_rope(q, pos_ids)
        k = self._apply_rope(k, pos_ids)

        attn_out = self._attention(q=q, k=k, v=v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.hidden_dim)

        x = x + self.resid_dropout(self.o_proj(attn_out))
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_clean_with_cache(
        self,
        x_clean: torch.Tensor,
        clean_pos: torch.Tensor,
        clean_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_clean: [B,Lc,D], clean_mask: [B,Lc,Lc]
        h = self.norm1(x_clean)
        q, k, v = self._project_qkv(h)
        q = self._apply_rope(q, clean_pos)
        k = self._apply_rope(k, clean_pos)

        attn_out = self._attention(q=q, k=k, v=v, attn_mask=clean_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(x_clean.size(0), x_clean.size(1), self.hidden_dim)

        x_clean = x_clean + self.resid_dropout(self.o_proj(attn_out))
        x_clean = x_clean + self.ffn(self.norm2(x_clean))
        return x_clean, k, v

    def forward_target_with_clean_cache(
        self,
        x_target: torch.Tensor,
        target_pos: torch.Tensor,
        kh: torch.Tensor,
        vh: torch.Tensor,
        target_kv_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # x_target: [B,K,D], kh/vh: [B,H,Lc,Dh], target_kv_mask: [B,K,Lc+K]
        h = self.norm1(x_target)
        qt, kt, vt = self._project_qkv(h)
        qt = self._apply_rope(qt, target_pos)
        kt = self._apply_rope(kt, target_pos)

        k_all = torch.cat([kh, kt], dim=2)
        v_all = torch.cat([vh, vt], dim=2)

        attn_out = self._attention(q=qt, k=k_all, v=v_all, attn_mask=target_kv_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(x_target.size(0), x_target.size(1), self.hidden_dim)

        x_target = x_target + self.resid_dropout(self.o_proj(attn_out))
        x_target = x_target + self.ffn(self.norm2(x_target))
        return x_target

    def forward_target_with_clean_cache_beam(
        self,
        x_target: torch.Tensor,
        target_pos: torch.Tensor,
        kh: torch.Tensor,
        vh: torch.Tensor,
        target_kv_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # x_target: [S,LB,K,D], kh/vh: [S,H,Lc,Dh], target_kv_mask: [K,Lc+K] or [S,LB,K,Lc+K]
        if x_target.dim() != 4:
            raise ValueError(f"x_target must be [S,LB,K,D], got {tuple(x_target.shape)}")
        s_batch, local_beam, k_tok, _ = x_target.shape
        if kh.dim() != 4 or vh.dim() != 4:
            raise ValueError(f"kh/vh must be [S,H,Lc,Dh], got {tuple(kh.shape)} / {tuple(vh.shape)}")
        if kh.size(0) != s_batch or vh.size(0) != s_batch:
            raise ValueError(f"cache batch mismatch: x_target={s_batch}, kh={kh.size(0)}, vh={vh.size(0)}")

        h = self.norm1(x_target)
        qt = self.q_proj(h).view(s_batch, local_beam, k_tok, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        kt = self.k_proj(h).view(s_batch, local_beam, k_tok, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        vt = self.v_proj(h).view(s_batch, local_beam, k_tok, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        qt_flat = qt.reshape(s_batch * local_beam, self.num_heads, k_tok, self.head_dim)
        kt_flat = kt.reshape(s_batch * local_beam, self.num_heads, k_tok, self.head_dim)
        qt_flat = self._apply_rope(qt_flat, target_pos)
        kt_flat = self._apply_rope(kt_flat, target_pos)
        qt = qt_flat.view(s_batch, local_beam, self.num_heads, k_tok, self.head_dim)
        kt = kt_flat.view(s_batch, local_beam, self.num_heads, k_tok, self.head_dim)

        if target_kv_mask is None:
            attn_mask = None
        elif target_kv_mask.dim() == 2:
            attn_mask = target_kv_mask.unsqueeze(0).unsqueeze(0)
        elif target_kv_mask.dim() == 3:
            attn_mask = target_kv_mask.unsqueeze(1)
        elif target_kv_mask.dim() == 4:
            attn_mask = target_kv_mask
        else:
            raise ValueError(
                f"target_kv_mask must be [K,L], [S,K,L], or [S,LB,K,L], got {tuple(target_kv_mask.shape)}"
            )

        # Flatten the local beam dimension so the beam path uses the same SDPA
        # implementation as the regular generation path.
        l_clean = int(kh.size(2))
        k_all = torch.cat(
            [
                kh.unsqueeze(1).expand(-1, local_beam, -1, -1, -1),
                kt,
            ],
            dim=3,
        )
        v_all = torch.cat(
            [
                vh.unsqueeze(1).expand(-1, local_beam, -1, -1, -1),
                vt,
            ],
            dim=3,
        )
        q_flat = qt.reshape(s_batch * local_beam, self.num_heads, k_tok, self.head_dim)
        k_flat = k_all.reshape(s_batch * local_beam, self.num_heads, l_clean + k_tok, self.head_dim)
        v_flat = v_all.reshape(s_batch * local_beam, self.num_heads, l_clean + k_tok, self.head_dim)
        if attn_mask is None:
            sdpa_attn_mask = None
        else:
            if attn_mask.size(0) == 1:
                sdpa_mask = attn_mask
            elif attn_mask.size(0) == s_batch:
                sdpa_mask = attn_mask.unsqueeze(1).expand(-1, local_beam, -1, -1).reshape(
                    s_batch * local_beam, 1, k_tok, l_clean + k_tok
                )
            else:
                sdpa_mask = attn_mask.reshape(s_batch * local_beam, 1, k_tok, l_clean + k_tok)
            sdpa_attn_mask = sdpa_mask.squeeze(1)
        attn_out = self._attention(
            q=q_flat,
            k=k_flat,
            v=v_flat,
            attn_mask=sdpa_attn_mask,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            s_batch, local_beam, k_tok, self.hidden_dim
        )

        x_target = x_target + self.resid_dropout(self.o_proj(attn_out))
        x_target = x_target + self.ffn(self.norm2(x_target))
        return x_target

    def append_clean_token(
        self,
        x_new: torch.Tensor,
        new_pos: torch.Tensor,
        kh: torch.Tensor,
        vh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_new: [B,1,D], kh/vh: [B,H,Lc,Dh]
        h = self.norm1(x_new)
        qn, kn, vn = self._project_qkv(h)
        qn = self._apply_rope(qn, new_pos)
        kn = self._apply_rope(kn, new_pos)

        k_all = torch.cat([kh, kn], dim=2)
        v_all = torch.cat([vh, vn], dim=2)

        attn_out = self._attention(q=qn, k=k_all, v=v_all, attn_mask=None)
        bsz = x_new.size(0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, 1, self.hidden_dim)

        x_new = x_new + self.resid_dropout(self.o_proj(attn_out))
        x_new = x_new + self.ffn(self.norm2(x_new))
        return x_new, kn, vn


class AsymmetricBlockDiffusionModel(nn.Module):
    """Dual-stream block diffusion model."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_rq_layers: int,
        num_pq_splits: int,
        vocab_global: int,
        ffn_dim: int = 0,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
        train_sdp_backend: str = "auto",
        eval_sdp_backend: str = "math",
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.ffn_dim = 4 * self.hidden_dim if int(ffn_dim) <= 0 else int(ffn_dim)
        self.num_rq_layers = int(num_rq_layers)  # B
        self.num_pq_splits = int(num_pq_splits)  # K
        self.vocab_global = int(vocab_global)
        self.num_clean_layers = self.num_rq_layers
        self.pad_token_id = 0
        self.mask_token_id = 1
        self.target_offset = 2
        self.train_sdp_backend = _normalize_sdp_backend(train_sdp_backend)
        self.eval_sdp_backend = _normalize_sdp_backend(eval_sdp_backend)

        self.target_embed = nn.Embedding(self.vocab_global + self.target_offset, self.hidden_dim, padding_idx=0)

        # Clean aggregation uses parameter-free mean pooling over the K slot embeddings.

        # Layer PE is only for real clean/target SID layers [0, B-1].
        self.layer_pe = nn.Embedding(self.num_rq_layers, self.hidden_dim)
        self.slice_pe = nn.Embedding(self.num_pq_splits, self.hidden_dim)
        self.clean_type_pe = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                BlockDiffusionLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=dropout,
                    rope_base=rope_base,
                    rope_cache_max_pos=4096,
                    train_sdp_backend=self.train_sdp_backend,
                    eval_sdp_backend=self.eval_sdp_backend,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_global, bias=False)

        self._index_cache: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
        self._base_mask_cache: Dict[Tuple[int, str], torch.Tensor] = {}
        self._bidir_base_mask_cache: Dict[Tuple[int, str], torch.Tensor] = {}
        self._gen_hidden_static_cache: Dict[Tuple[str, int, int, int, int], Dict[str, torch.Tensor]] = {}
        self._last_clean_hidden: Optional[torch.Tensor] = None

    def _build_indices(self, max_n: int, device: torch.device) -> Dict[str, torch.Tensor]:
        key = (int(max_n), str(device))
        if key in self._index_cache:
            return self._index_cache[key]

        b_clean = self.num_clean_layers
        b_target = self.num_rq_layers
        k = self.num_pq_splits

        clean_item = torch.arange(max_n, device=device, dtype=torch.long).repeat_interleave(b_clean)
        clean_layer = torch.arange(b_clean, device=device, dtype=torch.long).repeat(max_n)

        one_item_layers = torch.arange(b_target, device=device, dtype=torch.long).repeat_interleave(k)
        one_item_slices = torch.arange(k, device=device, dtype=torch.long).repeat(b_target)
        target_item = torch.arange(max_n, device=device, dtype=torch.long).repeat_interleave(b_target * k)
        target_layer = one_item_layers.repeat(max_n)
        target_slice = one_item_slices.repeat(max_n)

        full_item_pos = torch.cat([clean_item, target_item], dim=0)

        out = {
            "clean_item": clean_item,
            "clean_layer": clean_layer,
            "target_item": target_item,
            "target_layer": target_layer,
            "target_slice": target_slice,
            "full_item_pos": full_item_pos,
        }
        # Keep cache training-safe: do not store inference-mode tensors.
        if not torch.is_inference_mode_enabled():
            self._index_cache[key] = out
        return out

    def _build_base_mask(self, max_n: int, device: torch.device) -> torch.Tensor:
        key = (int(max_n), str(device))
        if key in self._base_mask_cache:
            return self._base_mask_cache[key]

        idx = self._build_indices(max_n=max_n, device=device)
        ci, cl = idx["clean_item"], idx["clean_layer"]
        ti, tl = idx["target_item"], idx["target_layer"]

        lc = ci.numel()
        lt = ti.numel()

        # clean -> clean (causal by item then layer)
        cc = (ci[None, :] < ci[:, None]) | ((ci[None, :] == ci[:, None]) & (cl[None, :] <= cl[:, None]))

        # clean -> target forbidden
        ct = torch.zeros((lc, lt), dtype=torch.bool, device=device)

        # target -> clean only strict previous clean blocks
        tc = (ci[None, :] < ti[:, None]) | ((ci[None, :] == ti[:, None]) & (cl[None, :] < tl[:, None]))

        # target -> target only same item and same layer
        tt = (ti[:, None] == ti[None, :]) & (tl[:, None] == tl[None, :])

        top = torch.cat([cc, ct], dim=1)
        bot = torch.cat([tc, tt], dim=1)
        base = torch.cat([top, bot], dim=0)  # [L,L]

        if not torch.is_inference_mode_enabled():
            self._base_mask_cache[key] = base
        return base

    def _build_bidir_base_mask(self, max_n: int, device: torch.device) -> torch.Tensor:
        key = (int(max_n), str(device))
        if key in self._bidir_base_mask_cache:
            return self._bidir_base_mask_cache[key]

        idx = self._build_indices(max_n=max_n, device=device)
        ci, cl = idx["clean_item"], idx["clean_layer"]
        ti, tl = idx["target_item"], idx["target_layer"]

        lc = ci.numel()
        lt = ti.numel()

        # clean -> clean: all layers from other items + current/previous layers from same item.
        # Only later clean layers from the same item remain hidden.
        cc = (ci[None, :] != ci[:, None]) | ((ci[None, :] == ci[:, None]) & (cl[None, :] <= cl[:, None]))

        # clean -> target forbidden
        ct = torch.zeros((lc, lt), dtype=torch.bool, device=device)

        # target -> clean: all layers from other items + previous layers from same item
        tc = (ci[None, :] != ti[:, None]) | ((ci[None, :] == ti[:, None]) & (cl[None, :] < tl[:, None]))

        # target -> target: only current item + current RQ layer slots
        tt = (ti[:, None] == ti[None, :]) & (tl[:, None] == tl[None, :])

        top = torch.cat([cc, ct], dim=1)
        bot = torch.cat([tc, tt], dim=1)
        base = torch.cat([top, bot], dim=0)  # [L,L]

        if not torch.is_inference_mode_enabled():
            self._bidir_base_mask_cache[key] = base
        return base

    def build_attention_mask(self, item_mask: torch.Tensor, attn_mode: str = "causal") -> torch.Tensor:
        """
        Build dual-stream mask.

        Args:
            item_mask: [B, n] bool, valid items per sample
            attn_mode: "causal", "bidir", or legacy "bidirectional"
        Returns:
            full_mask: [B, L_total, L_total] bool, True=visible
        """
        if item_mask.dim() != 2:
            raise ValueError(f"item_mask must be [B,n], got {tuple(item_mask.shape)}")
        _, max_n = item_mask.shape
        device = item_mask.device

        clean_valid = item_mask.repeat_interleave(self.num_clean_layers, dim=1)
        target_valid = item_mask.repeat_interleave(self.num_rq_layers * self.num_pq_splits, dim=1)
        token_valid = torch.cat([clean_valid, target_valid], dim=1)  # [B,L]
        if attn_mode == "causal":
            base = self._build_base_mask(max_n=max_n, device=device)
            full = base.unsqueeze(0) & token_valid[:, :, None] & token_valid[:, None, :]
        elif attn_mode == "bidir":
            base = self._build_bidir_base_mask(max_n=max_n, device=device)
            full = base.unsqueeze(0) & token_valid[:, :, None] & token_valid[:, None, :]
        elif attn_mode == "bidirectional":
            full = token_valid[:, :, None] & token_valid[:, None, :]
        else:
            raise ValueError(f"Unknown attn_mode={attn_mode}, expected 'causal'/'bidir'/'bidirectional'")

        l_total = token_valid.size(1)
        diag = torch.arange(l_total, device=device)
        empty_valid = token_valid & (~full.any(dim=2))
        full[:, diag, diag] |= (~token_valid) | empty_valid
        return full

    def _to_token_ids(self, ids: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert global ids to embedding token ids.

        Input ids use:
          - valid global id in [0, V_global-1]
          - masked id as -1
        Output token ids use:
          - PAD=0
          - MASK=1
          - real global id shifted by +2
        """
        if ids.shape != valid_mask.shape:
            raise ValueError(f"ids shape {tuple(ids.shape)} != valid_mask shape {tuple(valid_mask.shape)}")

        out = ids + self.target_offset
        out = out.to(torch.long)

        masked = ids < 0
        out[masked] = self.mask_token_id
        out[~valid_mask] = self.pad_token_id

        # NOTE: bound checks removed from hot path for decode speed.
        return out

    def _aggregate_clean_tokens(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool K PQ token embeddings into one clean token per RQ layer.

        Args:
            emb: [B,n,B_rq,K,D]
        Returns:
            clean: [B,n,B_rq,D]
        """
        return emb.mean(dim=3)

    def _aggregate_one_generated_layer(self, emb: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Mean-pool one generated target layer K embeddings into one clean token.

        Args:
            emb: [S,K,D]
            layer_idx: int in [0, B_rq-1]
        Returns:
            agg: [S,D]
        """
        if layer_idx < 0 or layer_idx >= self.num_rq_layers:
            raise ValueError(f"layer_idx out of range: {layer_idx}")
        return emb.mean(dim=1)

    def _embed_clean(
        self,
        clean_sid_ids: torch.Tensor,
        item_mask: torch.Tensor,
        idx: Dict[str, torch.Tensor],
    ):
        # clean_sid_ids: [B,n,B_rq,K_pq], item_mask: [B,n]
        bsz, n, b_rq, k_pq = clean_sid_ids.shape
        if b_rq != self.num_rq_layers or k_pq != self.num_pq_splits:
            raise ValueError("SID shape mismatch with model B/K")

        valid_ids = item_mask[:, :, None, None].expand(-1, -1, b_rq, k_pq)
        token_ids = self._to_token_ids(clean_sid_ids, valid_ids)

        emb = self.target_embed(token_ids)  # [B,n,B,K,D]
        clean_sid = self._aggregate_clean_tokens(emb)  # [B,n,B,D]
        sid_layer_idx = torch.arange(self.num_rq_layers, device=clean_sid.device, dtype=torch.long)
        clean_sid = clean_sid + self.layer_pe(sid_layer_idx).view(1, 1, self.num_rq_layers, self.hidden_dim)
        clean_sid = clean_sid + self.clean_type_pe.view(1, 1, 1, self.hidden_dim)

        clean = clean_sid.reshape(bsz, n * self.num_clean_layers, self.hidden_dim)

        clean_valid = item_mask.repeat_interleave(self.num_clean_layers, dim=1)
        clean = clean * clean_valid.unsqueeze(-1).to(clean.dtype)
        return clean, clean_valid

    def _embed_target(self, noisy_target_ids: torch.Tensor, item_mask: torch.Tensor, idx: Dict[str, torch.Tensor]):
        # noisy_target_ids: [B,n,B,K], item_mask: [B,n]
        bsz, n, b_rq, k_pq = noisy_target_ids.shape
        if b_rq != self.num_rq_layers or k_pq != self.num_pq_splits:
            raise ValueError("SID shape mismatch with model B/K")

        valid_ids = item_mask[:, :, None, None].expand(-1, -1, b_rq, k_pq)
        token_ids = self._to_token_ids(noisy_target_ids, valid_ids)

        target = self.target_embed(token_ids).reshape(bsz, n * b_rq * k_pq, self.hidden_dim)

        target = target + self.layer_pe(idx["target_layer"]).unsqueeze(0)
        target = target + self.slice_pe(idx["target_slice"]).unsqueeze(0)

        target_valid = item_mask.repeat_interleave(self.num_rq_layers * self.num_pq_splits, dim=1)
        target = target * target_valid.unsqueeze(-1).to(target.dtype)
        return target, target_valid

    def forward(
        self,
        sid_ids: torch.Tensor,
        item_mask: torch.Tensor,
        noisy_target_ids: Optional[torch.Tensor] = None,
        attn_mode: str = "causal",
        return_clean_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            sid_ids: [B,n,B_rq,K_pq], clean real SID ids
            item_mask: [B,n] bool
            noisy_target_ids: [B,n,B_rq,K_pq], with -1 for MASK; if None, use sid_ids
            attn_mode: "causal", "bidir", or legacy "bidirectional"
            return_clean_hidden: whether to cache final clean hidden states [B,n,B,D]

        Returns:
            logits: [B,n,B_rq,K_pq,V_global]
        """
        if sid_ids.dim() != 4:
            raise ValueError(f"sid_ids must be [B,n,B,K], got {tuple(sid_ids.shape)}")
        if item_mask.dim() != 2:
            raise ValueError(f"item_mask must be [B,n], got {tuple(item_mask.shape)}")

        bsz, max_n, b_rq, k_pq = sid_ids.shape
        if item_mask.shape[0] != bsz or item_mask.shape[1] != max_n:
            raise ValueError("item_mask shape mismatch")
        if b_rq != self.num_rq_layers or k_pq != self.num_pq_splits:
            raise ValueError("sid_ids B/K mismatch with model config")

        if noisy_target_ids is None:
            noisy_target_ids = sid_ids
        if noisy_target_ids.shape != sid_ids.shape:
            raise ValueError("noisy_target_ids shape mismatch")

        idx = self._build_indices(max_n=max_n, device=sid_ids.device)

        x_clean, _ = self._embed_clean(clean_sid_ids=sid_ids, item_mask=item_mask, idx=idx)
        x_target, _ = self._embed_target(noisy_target_ids=noisy_target_ids, item_mask=item_mask, idx=idx)

        x = torch.cat([x_clean, x_target], dim=1)
        x = self.emb_dropout(x)

        full_mask = self.build_attention_mask(item_mask=item_mask, attn_mode=attn_mode)
        full_pos = idx["full_item_pos"]

        for layer in self.layers:
            x = layer.forward_full(x=x, pos_ids=full_pos, attn_mask=full_mask)

        x = self.final_norm(x)
        l_clean = max_n * self.num_clean_layers
        clean_out = x[:, :l_clean, :].view(bsz, max_n, self.num_clean_layers, self.hidden_dim)
        target_out = x[:, l_clean:, :]

        logits = self.lm_head(target_out)
        logits = logits.view(bsz, max_n, self.num_rq_layers, self.num_pq_splits, self.vocab_global)
        self._last_clean_hidden = clean_out if return_clean_hidden else None
        return logits

    def pop_last_clean_hidden(self) -> Optional[torch.Tensor]:
        out = self._last_clean_hidden
        self._last_clean_hidden = None
        return out

    def _build_clean_cache(
        self,
        history_sid_ids: torch.Tensor,
    ) -> CleanKVCache:
        """
        Build clean KV cache from history sequence.

        Args:
            history_sid_ids: [bsz,n_hist,B,K] or [n_hist,B,K]
        """
        if history_sid_ids.dim() == 3:
            history_sid_ids = history_sid_ids.unsqueeze(0)
        if history_sid_ids.dim() != 4:
            raise ValueError(
                f"history_sid_ids must be [bsz,n_hist,B,K] or [n_hist,B,K], got {tuple(history_sid_ids.shape)}"
            )

        bsz, n_hist, b_rq, k_pq = history_sid_ids.shape
        if b_rq != self.num_rq_layers or k_pq != self.num_pq_splits:
            raise ValueError("history_sid_ids B/K mismatch")

        device = history_sid_ids.device

        item_mask = torch.ones((bsz, n_hist), dtype=torch.bool, device=device)
        idx = self._build_indices(max_n=n_hist, device=device)

        x_clean, _ = self._embed_clean(
            clean_sid_ids=history_sid_ids,
            item_mask=item_mask,
            idx=idx,
        )
        x_clean = self.emb_dropout(x_clean)

        l_clean = n_hist * self.num_clean_layers
        clean_pos = idx["clean_item"]
        clean_mask = self._build_base_mask(max_n=n_hist, device=device)[:l_clean, :l_clean]
        clean_mask = clean_mask.unsqueeze(0).expand(bsz, -1, -1).contiguous()

        k_per_layer: List[torch.Tensor] = []
        v_per_layer: List[torch.Tensor] = []

        for layer in self.layers:
            x_clean, kh, vh = layer.forward_clean_with_cache(
                x_clean=x_clean,
                clean_pos=clean_pos,
                clean_mask=clean_mask,
            )
            k_per_layer.append(kh)
            v_per_layer.append(vh)

        return CleanKVCache(
            k_per_layer=k_per_layer,
            v_per_layer=v_per_layer,
            clean_item_idx=idx["clean_item"].clone(),
            clean_layer_idx=idx["clean_layer"].clone(),
        )

    def _predict_one_layer_hidden_beam(
        self,
        cache: CleanKVCache,
        current_layer_ids: torch.Tensor,
        item_idx: int,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Predict hidden states for one target layer (K tokens) conditioned on clean cache.

        Args:
            current_layer_ids: [S,LB,K], values in [0,V-1] or -1 for MASK
        Returns:
            hidden: [S,LB,K,D]
        """
        if current_layer_ids.dim() != 3 or current_layer_ids.size(2) != self.num_pq_splits:
            raise ValueError(
                f"current_layer_ids must be [S,LB,K], got {tuple(current_layer_ids.shape)} with K={self.num_pq_splits}"
            )

        device = current_layer_ids.device
        s_batch = int(current_layer_ids.size(0))

        static_key = (
            str(device),
            int(item_idx),
            int(layer_idx),
            int(id(cache.clean_item_idx)),
            int(id(cache.clean_layer_idx)),
        )
        static = self._gen_hidden_static_cache.get(static_key, None)
        if static is None:
            k = self.num_pq_splits
            cur_layer = torch.full((k,), int(layer_idx), device=device, dtype=torch.long)
            cur_slice = torch.arange(k, device=device, dtype=torch.long)
            target_pos = torch.full((k,), int(item_idx), device=device, dtype=torch.long)

            clean_allow = (cache.clean_item_idx < item_idx) | (
                (cache.clean_item_idx == item_idx) & (cache.clean_layer_idx < layer_idx)
            )
            l_clean = int(clean_allow.numel())
            target_self = torch.ones((k, k), dtype=torch.bool, device=device)
            mask_2d = torch.cat(
                [clean_allow.unsqueeze(0).expand(k, l_clean), target_self],
                dim=1,
            ).contiguous()
            if bool(mask_2d.all()):
                mask_2d = None

            static = {
                "cur_layer": cur_layer,
                "cur_slice": cur_slice,
                "target_pos": target_pos,
                "mask_2d": mask_2d,
            }
            if len(self._gen_hidden_static_cache) >= 4096:
                self._gen_hidden_static_cache.clear()
            self._gen_hidden_static_cache[static_key] = static

        valid = torch.ones_like(current_layer_ids, dtype=torch.bool, device=device)
        token_ids = self._to_token_ids(current_layer_ids, valid)
        x_cur = self.target_embed(token_ids)

        cur_layer = static["cur_layer"]
        cur_slice = static["cur_slice"]
        x_cur = x_cur + self.layer_pe(cur_layer).view(1, 1, self.num_pq_splits, self.hidden_dim)
        x_cur = x_cur + self.slice_pe(cur_slice).view(1, 1, self.num_pq_splits, self.hidden_dim)

        x_target = self.emb_dropout(x_cur)
        target_pos = static["target_pos"]
        mask_2d = static["mask_2d"]

        for i, layer in enumerate(self.layers):
            kh = cache.k_per_layer[i]
            vh = cache.v_per_layer[i]
            if kh.size(0) != s_batch or vh.size(0) != s_batch:
                if kh.size(0) == 1 and vh.size(0) == 1 and s_batch > 1:
                    kh = kh.expand(s_batch, -1, -1, -1)
                    vh = vh.expand(s_batch, -1, -1, -1)
                else:
                    raise ValueError(
                        f"Cache batch mismatch at layer={i}: cache={kh.size(0)} target={s_batch}"
                    )
            x_target = layer.forward_target_with_clean_cache_beam(
                x_target=x_target,
                target_pos=target_pos,
                kh=kh,
                vh=vh,
                target_kv_mask=mask_2d,
            )

        x_target = self.final_norm(x_target)
        return x_target

    def _predict_one_layer_hidden(
        self,
        cache: CleanKVCache,
        current_layer_ids: torch.Tensor,
        item_idx: int,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Predict hidden states for one target layer (K tokens) conditioned on clean cache.

        Args:
            current_layer_ids: [1,K], values in [0,V-1] or -1 for MASK
        Returns:
            hidden: [S,K,D]
        """
        if current_layer_ids.dim() != 2 or current_layer_ids.size(1) != self.num_pq_splits:
            raise ValueError(
                f"current_layer_ids must be [S,K], got {tuple(current_layer_ids.shape)} with K={self.num_pq_splits}"
            )

        device = current_layer_ids.device
        batch_size = int(current_layer_ids.size(0))
        k = self.num_pq_splits

        static_key = (
            str(device),
            int(item_idx),
            int(layer_idx),
            int(id(cache.clean_item_idx)),
            int(id(cache.clean_layer_idx)),
        )
        static = self._gen_hidden_static_cache.get(static_key, None)
        if static is None:
            cur_layer = torch.full((k,), int(layer_idx), device=device, dtype=torch.long)
            cur_slice = torch.arange(k, device=device, dtype=torch.long)
            target_pos = torch.full((k,), int(item_idx), device=device, dtype=torch.long)

            clean_allow = (cache.clean_item_idx < item_idx) | (
                (cache.clean_item_idx == item_idx) & (cache.clean_layer_idx < layer_idx)
            )  # [L_clean]
            l_clean = int(clean_allow.numel())
            target_self = torch.ones((k, k), dtype=torch.bool, device=device)
            mask_2d = torch.cat(
                [clean_allow.unsqueeze(0).expand(k, l_clean), target_self],
                dim=1,
            ).contiguous()  # [K, L_clean+K]
            if bool(mask_2d.all()):
                mask_2d = None

            static = {
                "cur_layer": cur_layer,
                "cur_slice": cur_slice,
                "target_pos": target_pos,
                "mask_2d": mask_2d,
            }
            if len(self._gen_hidden_static_cache) >= 4096:
                self._gen_hidden_static_cache.clear()
            self._gen_hidden_static_cache[static_key] = static

        valid = torch.ones_like(current_layer_ids, dtype=torch.bool, device=device)
        token_ids = self._to_token_ids(current_layer_ids, valid)
        x_cur = self.target_embed(token_ids)  # [S,K,D]

        cur_layer = static["cur_layer"]
        cur_slice = static["cur_slice"]
        x_cur = x_cur + self.layer_pe(cur_layer).unsqueeze(0)
        x_cur = x_cur + self.slice_pe(cur_slice).unsqueeze(0)

        x_target = x_cur
        x_target = self.emb_dropout(x_target)

        target_pos = static["target_pos"]
        mask_2d = static["mask_2d"]
        mask = None if mask_2d is None else mask_2d.unsqueeze(0).expand(batch_size, -1, -1)

        for i, layer in enumerate(self.layers):
            kh = cache.k_per_layer[i]
            vh = cache.v_per_layer[i]
            if kh.size(0) != batch_size or vh.size(0) != batch_size:
                if kh.size(0) == 1 and vh.size(0) == 1 and batch_size > 1:
                    kh = kh.expand(batch_size, -1, -1, -1)
                    vh = vh.expand(batch_size, -1, -1, -1)
                else:
                    raise ValueError(
                        f"Cache batch mismatch at layer={i}: cache={kh.size(0)} target={batch_size}"
                    )
            x_target = layer.forward_target_with_clean_cache(
                x_target=x_target,
                target_pos=target_pos,
                kh=kh,
                vh=vh,
                target_kv_mask=mask,
            )

        x_target = self.final_norm(x_target)
        return x_target

    def _predict_one_layer_logits(
        self,
        cache: CleanKVCache,
        current_layer_ids: torch.Tensor,
        item_idx: int,
        layer_idx: int,
    ) -> torch.Tensor:
        hidden = self._predict_one_layer_hidden(
            cache=cache,
            current_layer_ids=current_layer_ids,
            item_idx=item_idx,
            layer_idx=layer_idx,
        )
        logits = self.lm_head(hidden)
        return logits

    def _append_generated_layer(
        self,
        cache: CleanKVCache,
        layer_ids: torch.Tensor,
        item_idx: int,
        layer_idx: int,
    ) -> CleanKVCache:
        # layer_ids: [S,K], all finalized global ids
        if layer_ids.dim() != 2 or layer_ids.size(1) != self.num_pq_splits:
            raise ValueError("layer_ids must be [S,K]")
        if (layer_ids < 0).any():
            raise ValueError("layer_ids must be finalized (no -1)")

        device = layer_ids.device
        batch_size = int(layer_ids.size(0))

        token_ids = layer_ids + self.target_offset
        emb = self.target_embed(token_ids)  # [S,K,D]
        agg = self._aggregate_one_generated_layer(emb=emb, layer_idx=int(layer_idx)).unsqueeze(1)  # [S,1,D]

        lidx = torch.tensor([int(layer_idx)], device=device, dtype=torch.long)
        x_new = agg + self.layer_pe(lidx).view(1, 1, -1) + self.clean_type_pe  # broadcast on S
        x_new = self.emb_dropout(x_new)

        new_pos = torch.tensor([int(item_idx)], device=device, dtype=torch.long)

        new_k: List[torch.Tensor] = []
        new_v: List[torch.Tensor] = []

        for i, layer in enumerate(self.layers):
            kh = cache.k_per_layer[i]
            vh = cache.v_per_layer[i]
            if kh.size(0) != batch_size or vh.size(0) != batch_size:
                if kh.size(0) == 1 and vh.size(0) == 1 and batch_size > 1:
                    kh = kh.expand(batch_size, -1, -1, -1)
                    vh = vh.expand(batch_size, -1, -1, -1)
                else:
                    raise ValueError(
                        f"Cache batch mismatch at layer={i}: cache={kh.size(0)} target={batch_size}"
                    )
            x_new, kn, vn = layer.append_clean_token(
                x_new=x_new,
                new_pos=new_pos,
                kh=kh,
                vh=vh,
            )
            new_k.append(torch.cat([kh, kn], dim=2))
            new_v.append(torch.cat([vh, vn], dim=2))

        new_clean_item_idx = torch.cat(
            [cache.clean_item_idx, torch.tensor([int(item_idx)], device=device, dtype=torch.long)], dim=0
        )
        new_clean_layer_idx = torch.cat(
            [cache.clean_layer_idx, torch.tensor([int(layer_idx)], device=device, dtype=torch.long)], dim=0
        )
        return CleanKVCache(
            k_per_layer=new_k,
            v_per_layer=new_v,
            clean_item_idx=new_clean_item_idx,
            clean_layer_idx=new_clean_layer_idx,
        )

    def _append_generated_layer_from_parent(
        self,
        cache: CleanKVCache,
        parent_idx: torch.Tensor,
        layer_ids: torch.Tensor,
        item_idx: int,
        layer_idx: int,
    ) -> CleanKVCache:
        # parent_idx: [S_new], layer_ids: [S_new,K]
        if parent_idx.dim() != 1:
            raise ValueError(f"parent_idx must be 1D, got {tuple(parent_idx.shape)}")
        if layer_ids.dim() != 2 or layer_ids.size(1) != self.num_pq_splits:
            raise ValueError("layer_ids must be [S,K]")
        if int(layer_ids.size(0)) != int(parent_idx.numel()):
            raise ValueError(
                f"parent_idx length {int(parent_idx.numel())} != layer_ids batch {int(layer_ids.size(0))}"
            )
        if (layer_ids < 0).any():
            raise ValueError("layer_ids must be finalized (no -1)")

        device = layer_ids.device
        batch_size = int(layer_ids.size(0))
        parent_idx = parent_idx.to(device=device, dtype=torch.long)

        token_ids = layer_ids + self.target_offset
        emb = self.target_embed(token_ids)
        agg = self._aggregate_one_generated_layer(emb=emb, layer_idx=int(layer_idx)).unsqueeze(1)

        lidx = torch.tensor([int(layer_idx)], device=device, dtype=torch.long)
        x_new = agg + self.layer_pe(lidx).view(1, 1, -1) + self.clean_type_pe
        x_new = self.emb_dropout(x_new)

        new_pos = torch.tensor([int(item_idx)], device=device, dtype=torch.long)
        new_k: List[torch.Tensor] = []
        new_v: List[torch.Tensor] = []

        for i, layer in enumerate(self.layers):
            kh_src = cache.k_per_layer[i]
            vh_src = cache.v_per_layer[i]
            kh = kh_src.index_select(0, parent_idx)
            vh = vh_src.index_select(0, parent_idx)
            x_new, kn, vn = layer.append_clean_token(
                x_new=x_new,
                new_pos=new_pos,
                kh=kh,
                vh=vh,
            )
            new_k.append(torch.cat([kh, kn], dim=2))
            new_v.append(torch.cat([vh, vn], dim=2))

        new_clean_item_idx = torch.cat(
            [cache.clean_item_idx, torch.tensor([int(item_idx)], device=device, dtype=torch.long)], dim=0
        )
        new_clean_layer_idx = torch.cat(
            [cache.clean_layer_idx, torch.tensor([int(layer_idx)], device=device, dtype=torch.long)], dim=0
        )
        return CleanKVCache(
            k_per_layer=new_k,
            v_per_layer=new_v,
            clean_item_idx=new_clean_item_idx,
            clean_layer_idx=new_clean_layer_idx,
        )

    @torch.no_grad()
    def build_generation_cache(
        self,
        history_sid_ids: torch.Tensor,
    ) -> CleanKVCache:
        """
        Public wrapper for generation cache build.
        """
        return self._build_clean_cache(history_sid_ids=history_sid_ids)

    @torch.no_grad()
    def predict_generation_layer_logits(
        self,
        cache: CleanKVCache,
        item_idx: int,
        layer_idx: int,
        current_layer_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Public wrapper for one-layer logits in generation.

        Args:
            cache: clean KV cache
            item_idx: target item index in macro sequence
            layer_idx: target layer index in [0, B-1]
            current_layer_ids: [S,K] global ids or -1 mask; if None, all mask
        Returns:
            logits: [S,K,V]
        """
        if current_layer_ids is None:
            dev = cache.clean_item_idx.device
            current_layer_ids = torch.full(
                (1, self.num_pq_splits),
                -1,
                dtype=torch.long,
                device=dev,
            )
        return self._predict_one_layer_logits(
            cache=cache,
            current_layer_ids=current_layer_ids,
            item_idx=item_idx,
            layer_idx=layer_idx,
        )

    @torch.no_grad()
    def predict_generation_layer_local_logits_beam(
        self,
        cache: CleanKVCache,
        item_idx: int,
        layer_idx: int,
        slot_codebook_sizes: List[int],
        slot_offsets: List[int],
        current_layer_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Public wrapper for one-layer local-vocab logits in generation with beam-shaped input.

        Args:
            cache: clean KV cache with batch=S
            item_idx: target item index in macro sequence
            layer_idx: target layer index in [0, B-1]
            slot_codebook_sizes: length B*K local vocab sizes
            slot_offsets: length B*K global offsets
            current_layer_ids: [S,LB,K] global ids or -1 mask; if None, all mask
        Returns:
            local_logits: [S,LB,K,V_local_max]
            local_offsets: [K]
            local_sizes: [K]
        """
        if current_layer_ids is None:
            dev = cache.clean_item_idx.device
            current_layer_ids = torch.full(
                (1, 1, self.num_pq_splits),
                -1,
                dtype=torch.long,
                device=dev,
            )
        s_total = int(self.num_rq_layers * self.num_pq_splits)
        if len(slot_codebook_sizes) != s_total:
            raise ValueError(
                f"slot_codebook_sizes length mismatch: got {len(slot_codebook_sizes)}, expected {s_total}"
            )
        if len(slot_offsets) != s_total:
            raise ValueError(f"slot_offsets length mismatch: got {len(slot_offsets)}, expected {s_total}")
        if not (0 <= int(layer_idx) < int(self.num_rq_layers)):
            raise ValueError(f"layer_idx out of range: {layer_idx} not in [0,{self.num_rq_layers})")

        hidden = self._predict_one_layer_hidden_beam(
            cache=cache,
            current_layer_ids=current_layer_ids,
            item_idx=item_idx,
            layer_idx=layer_idx,
        )

        base = int(layer_idx) * self.num_pq_splits
        offsets = [int(slot_offsets[base + k]) for k in range(self.num_pq_splits)]
        sizes = [int(slot_codebook_sizes[base + k]) for k in range(self.num_pq_splits)]
        if any(v <= 0 for v in sizes):
            raise ValueError(f"slot_codebook_sizes for layer={layer_idx} must be >0, got {sizes}")

        local_offsets = torch.tensor(offsets, device=hidden.device, dtype=torch.long)
        local_sizes = torch.tensor(sizes, device=hidden.device, dtype=torch.long)
        if all(sz == sizes[0] for sz in sizes) and all(offsets[k] == offsets[0] + k * sizes[0] for k in range(self.num_pq_splits)):
            local_vocab = int(sizes[0])
            weight_layer = self.lm_head.weight.narrow(0, int(offsets[0]), self.num_pq_splits * local_vocab)
            weight_layer = weight_layer.view(self.num_pq_splits, local_vocab, self.hidden_dim)
            local_logits = torch.einsum("slkd,kvd->slkv", hidden, weight_layer)
        else:
            local_vocab = int(max(sizes))
            neg_inf = torch.finfo(hidden.dtype).min
            local_logits = hidden.new_full((hidden.size(0), hidden.size(1), self.num_pq_splits, local_vocab), neg_inf)
            for k, (offset, size) in enumerate(zip(offsets, sizes)):
                weight_k = self.lm_head.weight.narrow(0, int(offset), int(size))
                logits_k = F.linear(hidden[:, :, k, :], weight_k)
                local_logits[:, :, k, : int(size)] = logits_k
        return local_logits, local_offsets, local_sizes

    @torch.no_grad()
    def predict_generation_layer_local_logits(
        self,
        cache: CleanKVCache,
        item_idx: int,
        layer_idx: int,
        slot_codebook_sizes: List[int],
        slot_offsets: List[int],
        current_layer_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Public wrapper for one-layer local-vocab logits in generation.

        Args:
            cache: clean KV cache
            item_idx: target item index in macro sequence
            layer_idx: target layer index in [0, B-1]
            slot_codebook_sizes: length B*K local vocab sizes
            slot_offsets: length B*K global offsets
            current_layer_ids: [S,K] global ids or -1 mask; if None, all mask
        Returns:
            local_logits: [S,K,V_local_max]
            local_offsets: [K]
            local_sizes: [K]
        """
        if current_layer_ids is None:
            dev = cache.clean_item_idx.device
            current_layer_ids = torch.full(
                (1, self.num_pq_splits),
                -1,
                dtype=torch.long,
                device=dev,
            )
        s_total = int(self.num_rq_layers * self.num_pq_splits)
        if len(slot_codebook_sizes) != s_total:
            raise ValueError(
                f"slot_codebook_sizes length mismatch: got {len(slot_codebook_sizes)}, expected {s_total}"
            )
        if len(slot_offsets) != s_total:
            raise ValueError(f"slot_offsets length mismatch: got {len(slot_offsets)}, expected {s_total}")
        if not (0 <= int(layer_idx) < int(self.num_rq_layers)):
            raise ValueError(f"layer_idx out of range: {layer_idx} not in [0,{self.num_rq_layers})")

        hidden = self._predict_one_layer_hidden(
            cache=cache,
            current_layer_ids=current_layer_ids,
            item_idx=item_idx,
            layer_idx=layer_idx,
        )

        base = int(layer_idx) * self.num_pq_splits
        offsets = [int(slot_offsets[base + k]) for k in range(self.num_pq_splits)]
        sizes = [int(slot_codebook_sizes[base + k]) for k in range(self.num_pq_splits)]
        if any(v <= 0 for v in sizes):
            raise ValueError(f"slot_codebook_sizes for layer={layer_idx} must be >0, got {sizes}")

        local_offsets = torch.tensor(offsets, device=hidden.device, dtype=torch.long)
        local_sizes = torch.tensor(sizes, device=hidden.device, dtype=torch.long)
        if all(sz == sizes[0] for sz in sizes) and all(offsets[k] == offsets[0] + k * sizes[0] for k in range(self.num_pq_splits)):
            local_vocab = int(sizes[0])
            weight_layer = self.lm_head.weight.narrow(0, int(offsets[0]), self.num_pq_splits * local_vocab)
            weight_layer = weight_layer.view(self.num_pq_splits, local_vocab, self.hidden_dim)
            local_logits = torch.einsum("skd,kvd->skv", hidden, weight_layer)
        else:
            local_vocab = int(max(sizes))
            neg_inf = torch.finfo(hidden.dtype).min
            local_logits = hidden.new_full((hidden.size(0), self.num_pq_splits, local_vocab), neg_inf)
            for k, (offset, size) in enumerate(zip(offsets, sizes)):
                weight_k = self.lm_head.weight.narrow(0, int(offset), int(size))
                logits_k = F.linear(hidden[:, k, :], weight_k)
                local_logits[:, k, : int(size)] = logits_k
        return local_logits, local_offsets, local_sizes

    @torch.no_grad()
    def predict_generation_layer_hidden(
        self,
        cache: CleanKVCache,
        item_idx: int,
        layer_idx: int,
        current_layer_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Public wrapper for one-layer hidden states in generation.

        Args:
            cache: clean KV cache
            item_idx: target item index in macro sequence
            layer_idx: target layer index in [0, B-1]
            current_layer_ids: [S,K] global ids or -1 mask; if None, all mask
        Returns:
            hidden: [S,K,D]
        """
        if current_layer_ids is None:
            dev = cache.clean_item_idx.device
            current_layer_ids = torch.full(
                (1, self.num_pq_splits),
                -1,
                dtype=torch.long,
                device=dev,
            )
        return self._predict_one_layer_hidden(
            cache=cache,
            current_layer_ids=current_layer_ids,
            item_idx=item_idx,
            layer_idx=layer_idx,
        )

    @torch.no_grad()
    def append_generation_layer(
        self,
        cache: CleanKVCache,
        layer_ids: torch.Tensor,
        item_idx: int,
        layer_idx: int,
    ) -> CleanKVCache:
        """
        Public wrapper to append one finalized generated layer into clean cache.

        Args:
            layer_ids: [K] or [S,K] global ids
        """
        if layer_ids.dim() == 1:
            layer_ids = layer_ids.unsqueeze(0)
        return self._append_generated_layer(
            cache=cache,
            layer_ids=layer_ids,
            item_idx=item_idx,
            layer_idx=layer_idx,
        )

    @torch.no_grad()
    def append_generation_layer_from_parent(
        self,
        cache: CleanKVCache,
        parent_idx: torch.Tensor,
        layer_ids: torch.Tensor,
        item_idx: int,
        layer_idx: int,
    ) -> CleanKVCache:
        """
        Public wrapper to append one finalized generated layer into clean cache,
        selecting parent states directly from a larger cache batch.

        Args:
            parent_idx: [S] indices into cache batch
            layer_ids: [K] or [S,K] global ids
        """
        if layer_ids.dim() == 1:
            layer_ids = layer_ids.unsqueeze(0)
        if parent_idx.dim() != 1:
            raise ValueError(f"parent_idx must be 1D, got {tuple(parent_idx.shape)}")
        return self._append_generated_layer_from_parent(
            cache=cache,
            parent_idx=parent_idx,
            layer_ids=layer_ids,
            item_idx=item_idx,
            layer_idx=layer_idx,
        )

    @torch.no_grad()
    def generate_next_item(
        self,
        history_sid_ids: torch.Tensor,
        diffusion_steps: int = 8,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Layer-by-layer generation for one next item.

        Args:
            history_sid_ids: [n_hist,B,K] or [1,n_hist,B,K]
            diffusion_steps: iterative denoising steps per layer
            temperature: softmax temperature for confidence ranking

        Returns:
            pred_sid: [B,K] global ids
        """
        if diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be > 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        if history_sid_ids.dim() == 4:
            if history_sid_ids.size(0) != 1:
                raise ValueError("generate_next_item currently supports batch=1")
            n_hist = int(history_sid_ids.size(1))
            dev = history_sid_ids.device
        elif history_sid_ids.dim() == 3:
            n_hist = int(history_sid_ids.size(0))
            dev = history_sid_ids.device
        else:
            raise ValueError(f"history_sid_ids must be [n,B,K] or [1,n,B,K], got {tuple(history_sid_ids.shape)}")

        cache = self._build_clean_cache(history_sid_ids=history_sid_ids)

        pred = torch.full(
            (1, self.num_rq_layers, self.num_pq_splits),
            -1,
            dtype=torch.long,
            device=dev,
        )

        item_idx = n_hist
        for layer_idx in range(self.num_rq_layers):
            layer_tokens = torch.full((1, self.num_pq_splits), -1, dtype=torch.long, device=dev)

            for _ in range(diffusion_steps):
                logits = self._predict_one_layer_logits(
                    cache=cache,
                    current_layer_ids=layer_tokens,
                    item_idx=item_idx,
                    layer_idx=layer_idx,
                )  # [1,K,V]
                probs = F.softmax(logits / float(temperature), dim=-1)

                masked_pos = (layer_tokens[0] < 0).nonzero(as_tuple=False).squeeze(-1)
                if masked_pos.numel() == 0:
                    break

                local_probs = probs[0, masked_pos, :]  # [M,V]
                conf, ids = local_probs.max(dim=-1)  # [M]
                best = int(conf.argmax().item())
                pos = int(masked_pos[best].item())
                tok = int(ids[best].item())
                layer_tokens[0, pos] = tok

            if (layer_tokens < 0).any():
                logits = self._predict_one_layer_logits(
                    cache=cache,
                    current_layer_ids=layer_tokens,
                    item_idx=item_idx,
                    layer_idx=layer_idx,
                )
                fill = logits.argmax(dim=-1)
                layer_tokens = torch.where(layer_tokens < 0, fill, layer_tokens)

            pred[:, layer_idx, :] = layer_tokens
            cache = self._append_generated_layer(
                cache=cache,
                layer_ids=layer_tokens,
                item_idx=item_idx,
                layer_idx=layer_idx,
            )

        return pred[0]


# Keep the public name used by the original release and its checkpoints.
SBDModel = AsymmetricBlockDiffusionModel


def sample_timestep_mask_ratio(
    batch_size: int,
    num_steps: int,
    device: Union[str, torch.device],
    min_ratio: float = 0.0,
    max_ratio: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample t ~ Uniform[1, num_steps], and map to ratio in [min_ratio, max_ratio].

    Returns:
      timesteps: [B]
      mask_ratio: [B]
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")
    if not (0.0 <= min_ratio <= 1.0 and 0.0 <= max_ratio <= 1.0 and min_ratio <= max_ratio):
        raise ValueError("Mask ratio bounds must satisfy 0 <= min <= max <= 1")

    t = torch.randint(1, num_steps + 1, (batch_size,), device=device)
    ratio = (t.to(torch.float32) / float(num_steps)).clamp(min=min_ratio, max=max_ratio)
    return t, ratio


def sample_groupwise_noise(
    clean_sid_ids: torch.Tensor,
    item_mask: torch.Tensor,
    mask_ratio: Union[float, torch.Tensor],
    mask_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply diffusion-style random masking on target stream.

    Args:
      clean_sid_ids: [B,n,B_rq,K_pq]
      item_mask: [B,n] bool
      mask_ratio: float scalar or [B] tensor

    Returns:
      noisy_target_ids: same shape as clean_sid_ids, masked positions set to mask_value
      loss_mask: bool same shape, True where prediction loss is applied
    """
    if clean_sid_ids.dim() != 4:
        raise ValueError(f"clean_sid_ids must be [B,n,B,K], got {tuple(clean_sid_ids.shape)}")
    if item_mask.dim() != 2:
        raise ValueError(f"item_mask must be [B,n], got {tuple(item_mask.shape)}")

    bsz, n, b_rq, k_pq = clean_sid_ids.shape
    if item_mask.shape != (bsz, n):
        raise ValueError("item_mask shape mismatch")

    if isinstance(mask_ratio, torch.Tensor):
        if mask_ratio.dim() != 1 or mask_ratio.size(0) != bsz:
            raise ValueError(f"mask_ratio tensor must be [B], got {tuple(mask_ratio.shape)}")
        ratio = mask_ratio.to(device=clean_sid_ids.device, dtype=torch.float32).clamp(0.0, 1.0)
    else:
        ratio = torch.full((bsz,), float(mask_ratio), device=clean_sid_ids.device, dtype=torch.float32).clamp(0.0, 1.0)

    valid = item_mask[:, :, None, None].expand(-1, -1, b_rq, k_pq)
    rnd = torch.rand_like(clean_sid_ids, dtype=torch.float32)
    loss_mask = (rnd < ratio[:, None, None, None]) & valid

    # Keep at least one masked token for each non-empty sample.
    valid_any = valid.reshape(bsz, -1).any(dim=1)
    has_any = loss_mask.reshape(bsz, -1).any(dim=1)
    need_fix = valid_any & (~has_any)
    if need_fix.any():
        bad = need_fix.nonzero(as_tuple=False).squeeze(-1)
        for i in bad.tolist():
            valid_idx = valid[i].reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            pick = int(valid_idx[torch.randint(0, valid_idx.numel(), (1,), device=clean_sid_ids.device)].item())
            loss_mask[i].reshape(-1)[pick] = True

    noisy = clean_sid_ids.clone()
    noisy[loss_mask] = int(mask_value)
    return noisy, loss_mask


def _resolve_slot_layout(
    b_rq: int,
    k_pq: int,
    v_global: int,
    device: torch.device,
    layer_codebook_sizes: List[int],
    slot_codebook_sizes: List[int],
    slot_offsets: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build per-slot layout for local-vocab CE.

    Returns:
      offsets_flat: [S]
      block_sizes_flat: [S] contiguous block length used for gather
      support_sizes_flat: [S] valid class count in each block
    where S = B*K.
    """
    s_total = int(b_rq * k_pq)
    if s_total <= 0:
        raise ValueError(f"Invalid slot count: B={b_rq}, K={k_pq}")

    if len(layer_codebook_sizes) != b_rq:
        raise ValueError(
            f"layer_codebook_sizes length mismatch: got {len(layer_codebook_sizes)}, expected {b_rq}"
        )
    if len(slot_codebook_sizes) != s_total:
        raise ValueError(
            f"slot_codebook_sizes length mismatch: got {len(slot_codebook_sizes)}, expected {s_total}"
        )
    if len(slot_offsets) != s_total:
        raise ValueError(
            f"slot_offsets length mismatch: got {len(slot_offsets)}, expected {s_total}"
        )

    layer_caps = [int(v) for v in layer_codebook_sizes]
    if any(v <= 0 for v in layer_caps):
        raise ValueError(f"layer_codebook_sizes must be positive, got {layer_codebook_sizes}")

    support_sizes_flat = torch.tensor(
        [layer_caps[b] for b in range(b_rq) for _ in range(k_pq)],
        dtype=torch.long,
        device=device,
    )
    block_sizes_flat = torch.tensor([int(v) for v in slot_codebook_sizes], dtype=torch.long, device=device)
    if torch.any(block_sizes_flat <= 0):
        raise ValueError(f"slot_codebook_sizes must be positive, got {slot_codebook_sizes}")
    if torch.any(support_sizes_flat <= 0):
        raise ValueError("support sizes must be positive")
    if torch.any(support_sizes_flat > block_sizes_flat):
        raise ValueError("support sizes cannot exceed slot block sizes")

    offsets_flat = torch.tensor([int(v) for v in slot_offsets], dtype=torch.long, device=device)

    if torch.any(offsets_flat < 0):
        raise ValueError("slot offsets must be >= 0")
    if torch.any(offsets_flat + block_sizes_flat > int(v_global)):
        raise ValueError(
            f"slot layout out of global vocab range: V_global={v_global}, "
            f"max(offset+block)={int((offsets_flat + block_sizes_flat).max().item())}"
        )

    return offsets_flat, block_sizes_flat, support_sizes_flat


def masked_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    layer_codebook_sizes: List[int],
    slot_codebook_sizes: List[int],
    slot_offsets: List[int],
) -> torch.Tensor:
    """
    Args:
      logits: [B,n,B_rq,K_pq,V]
      target_ids: [B,n,B_rq,K_pq]
      loss_mask: [B,n,B_rq,K_pq] bool
    """
    if logits.dim() != 5:
        raise ValueError(f"logits must be [B,n,B,K,V], got {tuple(logits.shape)}")
    if target_ids.shape != logits.shape[:-1]:
        raise ValueError("target_ids shape mismatch with logits")
    if loss_mask.shape != target_ids.shape:
        raise ValueError("loss_mask shape mismatch")
    bsz, n, b_rq, k_pq = target_ids.shape
    v_global = int(logits.size(-1))
    offsets_flat, block_sizes_flat, support_sizes_flat = _resolve_slot_layout(
        b_rq=b_rq,
        k_pq=k_pq,
        v_global=v_global,
        device=target_ids.device,
        layer_codebook_sizes=layer_codebook_sizes,
        slot_codebook_sizes=slot_codebook_sizes,
        slot_offsets=slot_offsets,
    )
    max_slot_v = int(block_sizes_flat.max().item())
    offsets = offsets_flat.view(1, 1, b_rq, k_pq)
    support_sizes_view = support_sizes_flat.view(1, 1, b_rq, k_pq)
    block_sizes_view = block_sizes_flat.view(1, 1, b_rq, k_pq)

    target_local = target_ids - offsets
    # Ignore padded/non-loss positions: only constrained targets participating in loss must be in-range.
    bad = ((target_local < 0) | (target_local >= support_sizes_view)) & loss_mask
    if torch.any(bad):
        raise ValueError("target_ids out of local vocab range on masked positions; check offset_rule / sid mapping")

    local_base = torch.arange(max_slot_v, device=target_ids.device, dtype=torch.long).view(1, 1, 1, 1, max_slot_v)
    gather_idx = offsets.unsqueeze(-1) + local_base
    gather_idx = gather_idx.expand(bsz, n, b_rq, k_pq, max_slot_v)
    local_logits = logits.gather(dim=-1, index=gather_idx)
    valid_cls = (local_base < support_sizes_view.unsqueeze(-1)) & (local_base < block_sizes_view.unsqueeze(-1))
    local_logits = local_logits.masked_fill(~valid_cls, torch.finfo(local_logits.dtype).min)

    masked_logits = local_logits[loss_mask]
    masked_targets = target_local[loss_mask]

    if masked_targets.numel() == 0:
        return logits.new_tensor(0.0, requires_grad=True)
    return F.cross_entropy(masked_logits, masked_targets)




def clean_alignment_cross_entropy_weighted(
    clean_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    item_mask: torch.Tensor,
    token_weights: torch.Tensor,
    lm_head_weight: torch.Tensor,
    layer_codebook_sizes: List[int],
    slot_codebook_sizes: List[int],
    slot_offsets: List[int],
) -> torch.Tensor:
    """
    Weighted clean-alignment CE.

    Each clean layer hidden [B,n,B,D] is projected through every slot-specific head of the
    corresponding RQ layer, using the same hidden vector for all K slot heads.
    """
    if clean_hidden.dim() != 4:
        raise ValueError(f"clean_hidden must be [B,n,B,D], got {tuple(clean_hidden.shape)}")
    if target_ids.dim() != 4:
        raise ValueError(f"target_ids must be [B,n,B,K], got {tuple(target_ids.shape)}")
    if item_mask.dim() != 2:
        raise ValueError(f"item_mask must be [B,n], got {tuple(item_mask.shape)}")
    if token_weights.shape != target_ids.shape:
        raise ValueError("token_weights shape mismatch")

    bsz, n, b_rq, hid = clean_hidden.shape
    if target_ids.shape[:3] != (bsz, n, b_rq):
        raise ValueError("target_ids shape mismatch with clean_hidden")
    k_pq = int(target_ids.size(3))
    if item_mask.shape != (bsz, n):
        raise ValueError("item_mask shape mismatch")
    if int(lm_head_weight.size(1)) != int(hid):
        raise ValueError("lm_head_weight hidden dim mismatch")

    _, block_sizes_flat, support_sizes_flat = _resolve_slot_layout(
        b_rq=b_rq,
        k_pq=k_pq,
        v_global=int(lm_head_weight.size(0)),
        device=clean_hidden.device,
        layer_codebook_sizes=layer_codebook_sizes,
        slot_codebook_sizes=slot_codebook_sizes,
        slot_offsets=slot_offsets,
    )
    offsets_flat = torch.tensor([int(v) for v in slot_offsets], dtype=torch.long, device=clean_hidden.device)

    item_valid_flat = item_mask.reshape(bsz * n)
    hidden_flat = clean_hidden.reshape(bsz * n, b_rq, hid)
    target_flat = target_ids.reshape(bsz * n, b_rq, k_pq)
    weight_flat = token_weights.reshape(bsz * n, b_rq, k_pq)

    total_loss = clean_hidden.new_zeros((), dtype=torch.float32)
    total_weight = clean_hidden.new_zeros((), dtype=torch.float32)
    neg_inf = None

    for layer_idx in range(b_rq):
        hidden_layer = hidden_flat[:, layer_idx, :]
        for slot_idx in range(k_pq):
            slot_flat_idx = int(layer_idx * k_pq + slot_idx)
            offset = int(offsets_flat[slot_flat_idx].item())
            block_size = int(block_sizes_flat[slot_flat_idx].item())
            support_size = int(support_sizes_flat[slot_flat_idx].item())
            pos_weights = weight_flat[:, layer_idx, slot_idx]
            valid_pos = item_valid_flat & (pos_weights > 0)
            if not bool(valid_pos.any()):
                continue

            logits = F.linear(hidden_layer[valid_pos], lm_head_weight.narrow(0, offset, block_size))
            if neg_inf is None:
                neg_inf = torch.finfo(logits.dtype).min
            if support_size < block_size:
                logits[:, support_size:] = neg_inf

            target_local = target_flat[:, layer_idx, slot_idx][valid_pos] - offset
            if torch.any((target_local < 0) | (target_local >= support_size)):
                raise ValueError("clean alignment target_ids out of local vocab range")

            ce = F.cross_entropy(logits, target_local, reduction="none")
            w = pos_weights[valid_pos].to(dtype=ce.dtype)
            total_loss = total_loss + (ce * w).sum().to(torch.float32)
            total_weight = total_weight + w.sum().to(torch.float32)

    if float(total_weight.item()) <= 0.0:
        return clean_hidden.new_tensor(0.0, requires_grad=True)
    return total_loss / total_weight.clamp_min(1e-12)

def masked_cross_entropy_weighted(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    token_weights: torch.Tensor,
    layer_codebook_sizes: List[int],
    slot_codebook_sizes: List[int],
    slot_offsets: List[int],
) -> torch.Tensor:
    """
    Weighted masked CE over target tokens.

    Args:
      logits: [B,n,B_rq,K_pq,V]
      target_ids: [B,n,B_rq,K_pq]
      loss_mask: [B,n,B_rq,K_pq] bool
      token_weights: [B,n,B_rq,K_pq] float, only masked positions are used
    """
    if logits.dim() != 5:
        raise ValueError(f"logits must be [B,n,B,K,V], got {tuple(logits.shape)}")
    if target_ids.shape != logits.shape[:-1]:
        raise ValueError("target_ids shape mismatch with logits")
    if loss_mask.shape != target_ids.shape:
        raise ValueError("loss_mask shape mismatch")
    if token_weights.shape != target_ids.shape:
        raise ValueError("token_weights shape mismatch")
    bsz, n, b_rq, k_pq = target_ids.shape
    v_global = int(logits.size(-1))
    offsets_flat, block_sizes_flat, support_sizes_flat = _resolve_slot_layout(
        b_rq=b_rq,
        k_pq=k_pq,
        v_global=v_global,
        device=target_ids.device,
        layer_codebook_sizes=layer_codebook_sizes,
        slot_codebook_sizes=slot_codebook_sizes,
        slot_offsets=slot_offsets,
    )
    max_slot_v = int(block_sizes_flat.max().item())
    offsets = offsets_flat.view(1, 1, b_rq, k_pq)
    support_sizes_view = support_sizes_flat.view(1, 1, b_rq, k_pq)
    block_sizes_view = block_sizes_flat.view(1, 1, b_rq, k_pq)

    target_local = target_ids - offsets
    bad = ((target_local < 0) | (target_local >= support_sizes_view)) & loss_mask
    if torch.any(bad):
        raise ValueError("target_ids out of local vocab range on masked positions; check offset_rule / sid mapping")

    local_base = torch.arange(max_slot_v, device=target_ids.device, dtype=torch.long).view(1, 1, 1, 1, max_slot_v)
    gather_idx = offsets.unsqueeze(-1) + local_base
    gather_idx = gather_idx.expand(bsz, n, b_rq, k_pq, max_slot_v)
    local_logits = logits.gather(dim=-1, index=gather_idx)
    valid_cls = (local_base < support_sizes_view.unsqueeze(-1)) & (local_base < block_sizes_view.unsqueeze(-1))
    local_logits = local_logits.masked_fill(~valid_cls, torch.finfo(local_logits.dtype).min)

    masked_logits = local_logits[loss_mask]
    masked_targets = target_local[loss_mask]
    masked_weights = token_weights[loss_mask]

    if masked_targets.numel() == 0:
        return logits.new_tensor(0.0, requires_grad=True)

    ce = F.cross_entropy(masked_logits, masked_targets, reduction="none")
    w = masked_weights.to(dtype=ce.dtype).clamp_min(0.0)
    denom = w.sum().clamp_min(1e-12)
    return (ce * w).sum() / denom
