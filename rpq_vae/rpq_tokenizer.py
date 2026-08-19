#!/usr/bin/env python3
"""RPQ-VAE tokenizer for Grid-SID generation."""

from __future__ import annotations

import math
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def load_asins_by_row(asin2row_json: str, n_rows: int) -> List[str]:
    asin2row = load_json(asin2row_json)
    if isinstance(asin2row, dict):
        items = list(asin2row.items())
    elif isinstance(asin2row, list):
        items = []
        for it in asin2row:
            if isinstance(it, dict) and "asin" in it and "row" in it:
                items.append((it["asin"], it["row"]))
            elif isinstance(it, (list, tuple)) and len(it) == 2:
                items.append((it[0], it[1]))
            else:
                raise ValueError(f"Unsupported asin2row item: {it}")
    else:
        raise ValueError(f"Unsupported asin2row format: {type(asin2row)}")

    rows = [int(r) for _, r in items]
    if len(rows) != len(set(rows)):
        raise ValueError("asin2row has duplicate row indices")
    if min(rows) != 0 or max(rows) != len(rows) - 1:
        raise ValueError("asin2row row indices must be contiguous 0..N-1")

    items.sort(key=lambda x: int(x[1]))
    asins = [asin for asin, _ in items]
    if len(asins) != n_rows:
        raise ValueError(f"asin count={len(asins)} != embedding rows={n_rows}")
    return asins


@dataclass
class RPQConfig:
    input_dim: int
    latent_dim: int
    num_layers: int
    num_splits: int
    codebook_size: int
    layer_codebook_sizes: Tuple[int, ...] | None = None
    hidden_dims: Tuple[int, ...] = (1024,)
    decoder_hidden_dims: Tuple[int, ...] | None = None
    dropout: float = 0.0
    beta_commit: float = 0.1
    normalized_target: bool = False
    use_ema: bool = True
    ema_decay: float = 0.995
    ema_eps: float = 1e-5
    dead_restart_threshold: float = 1.0
    dead_restart_noise: float = 0.03
    load_balance_loss: bool = True
    load_balance_weight: float = 1e-4
    load_balance_tau: float = 1.0
    load_balance_eps: float = 1e-8
    quant_balance_loss: bool = True
    quant_balance_weight: float = 1e-4
    diversity_loss: bool = True
    diversity_weight: float = 1e-2
    diversity_tau_init: float = 0.5
    diversity_tau_target: float = 0.1
    diversity_eps: float = 1e-8

    @property
    def split_dim(self) -> int:
        if self.latent_dim % self.num_splits != 0:
            raise ValueError(
                f"latent_dim={self.latent_dim} must be divisible by num_splits={self.num_splits}"
            )
        return self.latent_dim // self.num_splits

    @property
    def vocab_global(self) -> int:
        if self.layer_codebook_sizes is None:
            return int(self.num_layers * self.num_splits * self.codebook_size)
        return int(self.num_splits * sum(int(v) for v in self.layer_codebook_sizes))


class PureResidualPQ(nn.Module):

    def __init__(self, cfg: RPQConfig):
        super().__init__()
        self.cfg = cfg
        self.num_layers = int(cfg.num_layers)
        self.num_splits = int(cfg.num_splits)
        self.codebook_size = int(cfg.codebook_size)
        self.latent_dim = int(cfg.latent_dim)
        self.split_dim = int(cfg.split_dim)
        self.use_ema = bool(cfg.use_ema)
        self.ema_decay = float(cfg.ema_decay)
        self.ema_eps = float(cfg.ema_eps)
        self.dead_restart_threshold = float(cfg.dead_restart_threshold)
        self.dead_restart_noise = float(cfg.dead_restart_noise)
        self.load_balance_loss = bool(cfg.load_balance_loss)
        self.load_balance_weight = float(cfg.load_balance_weight)
        self.load_balance_tau = float(cfg.load_balance_tau)
        self.load_balance_eps = float(cfg.load_balance_eps)
        self.use_load_balance_loss = bool(self.load_balance_loss and self.load_balance_weight > 0.0)
        self.quant_balance_loss = bool(cfg.quant_balance_loss)
        self.quant_balance_weight = float(cfg.quant_balance_weight)
        self.use_quant_balance_loss = bool(
            self.quant_balance_loss and self.quant_balance_weight > 0.0
        )
        self.diversity_loss = bool(cfg.diversity_loss)
        self.diversity_weight = float(cfg.diversity_weight)
        self.diversity_tau_init = float(cfg.diversity_tau_init)
        self.diversity_tau_target = float(cfg.diversity_tau_target)
        self.diversity_eps = float(cfg.diversity_eps)
        self.use_diversity_loss = bool(self.diversity_loss and self.diversity_weight > 0.0)
        if self.load_balance_weight < 0.0:
            raise ValueError(f"load_balance_weight must be >=0, got {cfg.load_balance_weight}")
        if self.quant_balance_weight < 0.0:
            raise ValueError(f"quant_balance_weight must be >=0, got {cfg.quant_balance_weight}")
        if self.diversity_weight < 0.0:
            raise ValueError(f"diversity_weight must be >=0, got {cfg.diversity_weight}")
        if self.diversity_eps <= 0.0:
            raise ValueError(f"diversity_eps must be >0, got {cfg.diversity_eps}")
        if not (0.0 <= self.diversity_tau_init <= 1.0):
            raise ValueError(f"diversity_tau_init must be in [0,1], got {cfg.diversity_tau_init}")
        if not (0.0 <= self.diversity_tau_target <= 1.0):
            raise ValueError(f"diversity_tau_target must be in [0,1], got {cfg.diversity_tau_target}")
        if self.load_balance_tau <= 0.0:
            raise ValueError(f"load_balance_tau must be >0, got {cfg.load_balance_tau}")
        if self.load_balance_eps <= 0.0:
            raise ValueError(f"load_balance_eps must be >0, got {cfg.load_balance_eps}")
        if cfg.layer_codebook_sizes is None:
            layer_caps = [self.codebook_size] * self.num_layers
        else:
            layer_caps = [int(v) for v in cfg.layer_codebook_sizes]
            if len(layer_caps) != self.num_layers:
                raise ValueError(
                    f"layer_codebook_sizes length={len(layer_caps)} must equal num_layers={self.num_layers}"
                )
            if any(v <= 0 or v > self.codebook_size for v in layer_caps):
                raise ValueError(
                    f"Each layer codebook size must be in [1,{self.codebook_size}], got {layer_caps}"
                )
        self.register_buffer("layer_codebook_sizes", torch.tensor(layer_caps, dtype=torch.long))

        codebooks = torch.randn(
            self.num_layers, self.num_splits, self.codebook_size, self.split_dim
        ) * 0.02
        if self.use_ema:
            self.register_buffer("codebooks", codebooks)
            self.register_buffer(
                "ema_cluster_size",
                torch.zeros(self.num_layers, self.num_splits, self.codebook_size),
            )
            self.register_buffer(
                "ema_embed_sum",
                torch.zeros(
                    self.num_layers,
                    self.num_splits,
                    self.codebook_size,
                    self.split_dim,
                ),
            )
        else:
            self.codebooks = nn.Parameter(codebooks)  # type: ignore[assignment]

        offsets = []
        cur = 0
        for b in range(self.num_layers):
            row = []
            cap_b = int(layer_caps[b])
            for _ in range(self.num_splits):
                row.append(int(cur))
                cur += cap_b
            offsets.append(row)
        self.register_buffer("global_offsets", torch.tensor(offsets, dtype=torch.long))

        rot_init = torch.eye(self.latent_dim).unsqueeze(0).repeat(self.num_layers, 1, 1)
        self.rot_mats: nn.Parameter = nn.Parameter(rot_init)

    @staticmethod
    def _pairwise_l2(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        x2 = (x * x).sum(dim=-1, keepdim=True)
        c2 = (codebook * codebook).sum(dim=-1).unsqueeze(0)
        xc = x @ codebook.t()
        return x2 + c2 - 2.0 * xc

    def _diversity_tau_for_layer(self, layer_idx: int) -> float:
        if self.num_layers <= 1:
            return float(self.diversity_tau_init)
        t = float(layer_idx)
        t_max = float(self.num_layers - 1)
        return float(
            self.diversity_tau_target
            + 0.5
            * (self.diversity_tau_init - self.diversity_tau_target)
            * (1.0 + math.cos(math.pi * t / t_max))
        )

    @torch.no_grad()
    def _ema_update(self, b: int, k: int, x: torch.Tensor, idx: torch.Tensor, active_size: int | None = None) -> None:
        active = int(self.codebook_size if active_size is None else active_size)
        if active <= 0 or active > self.codebook_size:
            raise ValueError(f"Invalid active_size={active} for codebook_size={self.codebook_size}")

        counts = torch.bincount(idx, minlength=active).to(x.dtype)
        embed_sum = torch.zeros(active, self.split_dim, dtype=x.dtype, device=x.device)
        embed_sum.index_add_(0, idx, x)

        self.ema_cluster_size[b, k, :active].mul_(self.ema_decay).add_(
            counts * (1.0 - self.ema_decay)
        )
        self.ema_embed_sum[b, k, :active].mul_(self.ema_decay).add_(
            embed_sum * (1.0 - self.ema_decay)
        )

        denom = self.ema_cluster_size[b, k, :active].unsqueeze(-1) + self.ema_eps
        new_active = self.ema_embed_sum[b, k, :active] / denom

        dead = self.ema_cluster_size[b, k, :active] < self.dead_restart_threshold
        if dead.any():
            dead_idx = dead.nonzero(as_tuple=False).squeeze(1)
            if x.size(0) > 0:
                rand_rows = torch.randint(0, x.size(0), (dead_idx.numel(),), device=x.device)
                repl = x[rand_rows]
                if self.dead_restart_noise > 0:
                    repl = repl + self.dead_restart_noise * torch.randn_like(repl)
                new_active[dead_idx] = repl
                self.ema_cluster_size[b, k, :active][dead_idx] = 1.0
                self.ema_embed_sum[b, k, :active][dead_idx] = repl

        new_cb = self.codebooks[b, k].clone()
        new_cb[:active] = new_active
        self.codebooks[b, k].copy_(new_cb)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z.dim() != 2 or z.size(-1) != self.latent_dim:
            raise ValueError(f"Expected z=[N,{self.latent_dim}], got {tuple(z.shape)}")

        n = z.size(0)
        residual = z
        recon_sum_val = torch.zeros_like(z)
        local_idx_layers: List[torch.Tensor] = []
        commit_loss = z.new_tensor(0.0)
        vq_loss = z.new_tensor(0.0)
        load_balance_loss = z.new_tensor(0.0)
        quant_balance_loss = z.new_tensor(0.0)
        diversity_loss = z.new_tensor(0.0)

        usage_hits = torch.zeros(
            self.num_layers,
            self.num_splits,
            self.codebook_size,
            device=z.device,
            dtype=z.dtype,
        )

        for b in range(self.num_layers):
            active_v = int(self.layer_codebook_sizes[b].item())
            rot_b = self.rot_mats[b].to(device=z.device, dtype=z.dtype)
            residual_rot = residual @ rot_b

            residual_chunks = residual_rot.view(n, self.num_splits, self.split_dim)
            q_chunks = []
            idx_chunks = []
            split_errors = []
            if self.use_diversity_loss and self.num_splits > 1:
                tau_b = self._diversity_tau_for_layer(b)
                chunk_norm = residual_chunks.norm(dim=-1).clamp_min(self.diversity_eps)
                for i in range(self.num_splits):
                    x_i = residual_chunks[:, i, :]
                    for j in range(i + 1, self.num_splits):
                        x_j = residual_chunks[:, j, :]
                        cos_ij = (x_i * x_j).sum(dim=-1) / (chunk_norm[:, i] * chunk_norm[:, j])
                        pair_pen = F.relu(cos_ij.abs() - tau_b).mean()
                        diversity_loss = diversity_loss + 2.0 * pair_pen

            for k in range(self.num_splits):
                x = residual_chunks[:, k, :]
                cb = self.codebooks[b, k]
                dist = self._pairwise_l2(x, cb[:active_v])
                idx = dist.argmin(dim=-1)
                q = F.embedding(idx, cb[:active_v])
                if self.use_load_balance_loss:
                    cb_lb = cb[:active_v].detach().clone()
                    dist_lb = self._pairwise_l2(x, cb_lb)
                    probs = torch.softmax(-dist_lb / self.load_balance_tau, dim=-1)
                    q_hist = probs.mean(dim=0).clamp_min(self.load_balance_eps)
                    q_hist = q_hist / q_hist.sum()
                    load_balance_loss = load_balance_loss + (
                        q_hist * (torch.log(q_hist) + math.log(float(active_v)))
                    ).sum()
                if self.use_quant_balance_loss:
                    err = ((x - q) ** 2).sum(dim=-1).mean()
                    split_errors.append(err)

                if self.training and self.use_ema:
                    self._ema_update(b, k, x.detach(), idx, active_size=active_v)

                q_chunks.append(q)
                idx_chunks.append(idx)

                commit_loss = commit_loss + F.mse_loss(x, q.detach())
                vq_loss = vq_loss + F.mse_loss(x.detach(), q)
                usage_hits[b, k].scatter_add_(
                    0,
                    idx,
                    torch.ones_like(idx, dtype=usage_hits.dtype),
                )

            layer_q_rot = torch.stack(q_chunks, dim=1).reshape(n, self.latent_dim)
            layer_idx = torch.stack(idx_chunks, dim=1)
            local_idx_layers.append(layer_idx)

            layer_q = layer_q_rot @ rot_b.t()
            recon_sum_val = recon_sum_val + layer_q
            residual = residual - layer_q
            if self.use_quant_balance_loss:
                layer_errs = torch.stack(split_errors, dim=0)
                layer_err_mean = layer_errs.mean()
                quant_balance_loss = quant_balance_loss + ((layer_errs - layer_err_mean) ** 2).sum()

        recon_sum = z + (recon_sum_val - z).detach()

        local_ids = torch.stack(local_idx_layers, dim=1)
        global_ids = local_ids + self.global_offsets.unsqueeze(0)

        denom = float(max(1, self.num_layers * self.num_splits))
        commit_loss = commit_loss / denom
        vq_loss = vq_loss / denom
        load_balance_loss = load_balance_loss / denom
        quant_balance_loss = quant_balance_loss / denom
        pair_denom = float(max(1, self.num_splits * max(1, self.num_splits - 1)))
        diversity_loss = diversity_loss / float(max(1, self.num_layers)) / pair_denom

        return {
            "quantized": recon_sum,
            "residual": residual,
            "local_ids": local_ids,
            "global_ids": global_ids,
            "commit_loss": commit_loss,
            "vq_loss": vq_loss,
            "load_balance_loss": load_balance_loss,
            "quant_balance_loss": quant_balance_loss,
            "diversity_loss": diversity_loss,
            "usage_hits": usage_hits,
        }


class RPQTokenizer(nn.Module):
    def __init__(self, cfg: RPQConfig):
        super().__init__()
        self.cfg = cfg

        hidden_dims = tuple(int(h) for h in cfg.hidden_dims)
        if any(h <= 0 for h in hidden_dims):
            raise ValueError(f"hidden_dims must be positive integers, got {hidden_dims}")
        if cfg.decoder_hidden_dims is None:
            decoder_hidden_dims = hidden_dims
        else:
            decoder_hidden_dims = tuple(int(h) for h in cfg.decoder_hidden_dims)
            if any(h <= 0 for h in decoder_hidden_dims):
                raise ValueError(
                    f"decoder_hidden_dims must be positive integers, got {decoder_hidden_dims}"
                )

        self.encoder = self._build_encoder(
            input_dim=cfg.input_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=hidden_dims,
            dropout=cfg.dropout,
        )
        self.decoder = self._build_decoder(
            input_dim=cfg.input_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=decoder_hidden_dims,
            dropout=cfg.dropout,
        )

        self.quantizer = PureResidualPQ(cfg)

    @staticmethod
    def _build_encoder(
        input_dim: int,
        latent_dim: int,
        hidden_dims: Tuple[int, ...],
        dropout: float,
    ) -> nn.Module:
        dims = [int(input_dim), *list(hidden_dims), int(latent_dim)]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        layers.append(nn.LayerNorm(latent_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_decoder(
        input_dim: int,
        latent_dim: int,
        hidden_dims: Tuple[int, ...],
        dropout: float,
    ) -> nn.Module:
        rev_hidden = list(hidden_dims)[::-1]
        dims = [int(latent_dim), *rev_hidden, int(input_dim)]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        if self.cfg.normalized_target:
            x = F.normalize(x, p=2, dim=-1)
            x_hat = F.normalize(x_hat, p=2, dim=-1)
        return 1.0 - F.cosine_similarity(x_hat, x, dim=-1).mean()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        q_out = self.quantizer(z)
        z_q = q_out["quantized"]
        x_hat = self.decoder(z_q)

        rec = self.reconstruction_loss(x, x_hat)
        total = rec + self.cfg.beta_commit * q_out["commit_loss"]
        if self.cfg.load_balance_loss and self.cfg.load_balance_weight > 0.0:
            total = total + self.cfg.load_balance_weight * q_out["load_balance_loss"]
        if self.cfg.quant_balance_loss and self.cfg.quant_balance_weight > 0.0:
            total = total + self.cfg.quant_balance_weight * q_out["quant_balance_loss"]
        if self.cfg.diversity_loss and self.cfg.diversity_weight > 0.0:
            total = total + self.cfg.diversity_weight * q_out["diversity_loss"]
        if not self.quantizer.use_ema:
            total = total + q_out["vq_loss"]

        return {
            "loss": total,
            "recon_loss": rec,
            "vq_loss": q_out["vq_loss"],
            "commit_loss": q_out["commit_loss"],
            "load_balance_loss": q_out["load_balance_loss"],
            "quant_balance_loss": q_out["quant_balance_loss"],
            "diversity_loss": q_out["diversity_loss"],
            "x_hat": x_hat,
            "z": z,
            "z_q": z_q,
            "local_ids": q_out["local_ids"],
            "global_ids": q_out["global_ids"],
            "usage_hits": q_out["usage_hits"],
        }


@torch.no_grad()
def quantize_all(
    model: RPQTokenizer,
    embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    n = embeddings.size(0)
    all_local: List[np.ndarray] = []
    all_global: List[np.ndarray] = []
    for st in range(0, n, batch_size):
        ed = min(st + batch_size, n)
        xb = embeddings[st:ed].to(device)
        out = model(xb)
        all_local.append(out["local_ids"].cpu().numpy().astype(np.int64))
        all_global.append(out["global_ids"].cpu().numpy().astype(np.int64))
    return np.concatenate(all_local, axis=0), np.concatenate(all_global, axis=0)


def sid_usage_stats(local_ids: np.ndarray, codebook_size: int) -> Dict:
    n, b, k = local_ids.shape
    out = {
        "num_items": int(n),
        "num_layers": int(b),
        "num_splits": int(k),
        "codebook_size": int(codebook_size),
    }
    layer_stats = []
    for bi in range(b):
        split_stats = []
        for ki in range(k):
            vals = local_ids[:, bi, ki]
            hist = np.bincount(vals, minlength=codebook_size)
            used = int((hist > 0).sum())
            p = hist / max(1, int(hist.sum()))
            top1 = float(p.max()) if p.size else 0.0
            split_stats.append(
                {
                    "split": int(ki),
                    "used_codes": used,
                    "utilization": float(used / float(codebook_size)),
                    "top1_share": top1,
                }
            )
        layer_stats.append({"layer": int(bi), "splits": split_stats})
    out["layer_stats"] = layer_stats

    flat = local_ids.reshape(n, b * k)
    unique = int(np.unique(flat, axis=0).shape[0])
    out["unique_sid_count"] = unique
    out["unique_sid_ratio"] = float(unique / float(max(1, n)))
    return out
