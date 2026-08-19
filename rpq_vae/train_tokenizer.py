#!/usr/bin/env python3
"""Train RPQ-VAE and export Grid-SID."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from functools import lru_cache

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from rpq_tokenizer import (
    RPQConfig,
    RPQTokenizer,
    l2_normalize_np,
    load_asins_by_row,
    quantize_all,
    save_json,
    set_seed,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Video_Games")
    p.add_argument("--embeddings_npy", type=str, default="")
    p.add_argument("--asin2row_json", type=str, default="")
    p.add_argument("--out_dir", type=str, default="")

    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3, help="B")
    p.add_argument("--num_splits", type=int, default=4, help="K")
    p.add_argument("--codebook_size", type=int, default=1024, help="V")
    p.add_argument(
        "--layer_codebook_sizes",
        type=str,
        default="",
        help="Optional comma-separated per-layer effective vocab sizes, e.g. '16,16,8'.",
    )
    p.add_argument(
        "--hidden_dims",
        type=str,
        default="1024",
        help="Comma-separated hidden dims, e.g. '2048,1024'. Empty means direct linear projection to latent.",
    )
    p.add_argument(
        "--decoder_hidden_dims",
        type=str,
        default="",
        help="Comma-separated decoder hidden dims. Empty means mirror --hidden_dims.",
    )
    p.add_argument("--hidden_dim", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--beta_commit", type=float, default=0.1)
    p.add_argument(
        "--load_balance_loss",
        dest="load_balance_loss",
        action="store_true",
        help="Enable soft per-slot load-balancing loss KL(q||U) on assignment probabilities.",
    )
    p.add_argument(
        "--no_load_balance_loss",
        dest="load_balance_loss",
        action="store_false",
        help="Disable soft load-balancing loss.",
    )
    p.add_argument(
        "--load_balance_weight",
        type=float,
        default=1e-4,
        help="Weight for soft load-balancing loss.",
    )
    p.add_argument(
        "--load_balance_tau",
        type=float,
        default=1.0,
        help="Softmax temperature for load-balancing assignment probabilities.",
    )
    p.add_argument(
        "--load_balance_eps",
        type=float,
        default=1e-8,
        help="Numerical floor for load-balancing probability normalization.",
    )
    p.add_argument(
        "--quant_balance_loss",
        dest="quant_balance_loss",
        action="store_true",
        help="Enable per-layer quantization error balancing across K splits.",
    )
    p.add_argument(
        "--no_quant_balance_loss",
        dest="quant_balance_loss",
        action="store_false",
        help="Disable per-layer quantization error balancing loss.",
    )
    p.add_argument(
        "--quant_balance_weight",
        type=float,
        default=1e-4,
        help="Weight for quantization error balancing loss across splits.",
    )
    p.add_argument(
        "--diversity_loss",
        dest="diversity_loss",
        action="store_true",
        help="Enable soft diversity regularization across K split features in each layer.",
    )
    p.add_argument(
        "--no_diversity_loss",
        dest="diversity_loss",
        action="store_false",
        help="Disable soft diversity regularization.",
    )
    p.add_argument(
        "--diversity_weight",
        type=float,
        default=1e-2,
        help="Weight for soft diversity regularization.",
    )
    p.add_argument(
        "--diversity_tau_init",
        type=float,
        default=0.5,
        help="Layer-1 tolerance tau for diversity regularization.",
    )
    p.add_argument(
        "--diversity_tau_target",
        type=float,
        default=0.1,
        help="Final-layer tolerance tau for diversity regularization.",
    )
    p.add_argument(
        "--diversity_eps",
        type=float,
        default=1e-8,
        help="Numerical floor for diversity regularization normalization.",
    )

    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--ema_eps", type=float, default=1e-5)
    p.add_argument("--dead_restart_threshold", type=float, default=1.0)
    p.add_argument("--dead_restart_noise", type=float, default=0.03)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--normalize", dest="normalize", action="store_true")
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--normalize_input", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no_normalize_input", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--normalize_recon_target", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no_normalize_recon_target", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dedup_sid", dest="dedup_sid", action="store_true")
    p.add_argument("--no_dedup_sid", dest="dedup_sid", action="store_false")
    p.set_defaults(
        dedup_sid=True,
        normalize=None,
        load_balance_loss=True,
        quant_balance_loss=True,
        diversity_loss=True,
    )

    return p.parse_args()


def init_distributed(device_arg: str) -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        if device_arg.startswith("cuda") and not torch.cuda.is_available():
            return False, rank, world_size, local_rank, torch.device("cpu")
        return False, rank, world_size, local_rank, torch.device(device_arg)

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed mode requires CUDA.")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")


def parse_hidden_dims(hidden_dims_text: str) -> tuple[int, ...]:
    raw = (hidden_dims_text or "").strip()
    if raw == "" or raw.lower() in ("none", "null", "[]"):
        return ()
    if raw.startswith("["):
        vals = json.loads(raw)
        if not isinstance(vals, list):
            raise ValueError(f"hidden_dims JSON must be a list, got: {type(vals)}")
        dims = tuple(int(v) for v in vals)
    else:
        dims = tuple(int(v.strip()) for v in raw.split(",") if v.strip() != "")
    if any(d <= 0 for d in dims):
        raise ValueError(f"hidden_dims must be positive integers, got {dims}")
    return dims


def parse_layer_codebook_sizes(text: str, num_layers: int, codebook_size: int):
    raw = str(text or "").strip()
    if raw == "":
        return None
    vals = [int(v.strip()) for v in raw.split(",") if v.strip() != ""]
    if len(vals) != int(num_layers):
        raise ValueError(
            f"layer_codebook_sizes length={len(vals)} must equal num_layers={num_layers}"
        )
    if any(v <= 0 or v > int(codebook_size) for v in vals):
        raise ValueError(
            f"each layer_codebook_sizes entry must be in [1,{codebook_size}], got {vals}"
        )
    return tuple(vals)


def resolve_paths(args):
    base = f"data/{args.dataset}/item_embeddings"
    embeddings_npy = args.embeddings_npy or f"{base}/embeddings.npy"
    asin2row_json = args.asin2row_json or f"{base}/asin2row.json"
    out_dir = args.out_dir or f"outputs/{args.dataset}/grid_sid"
    return embeddings_npy, asin2row_json, out_dir


def _fmt_dur(sec: float) -> str:
    s = float(max(0.0, sec))
    if s < 60.0:
        return f"{s:.2f}s"
    if s < 3600.0:
        return f"{s / 60.0:.2f}m"
    return f"{s / 3600.0:.2f}h"


def run_epoch(
    model,
    loader,
    optimizer=None,
    scheduler=None,
    grad_clip: float = 1.0,
    device: str = "cuda",
    is_dist: bool = False,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    core_model = model.module if hasattr(model, "module") else model

    totals = {
        "loss": 0.0,
        "rec": 0.0,
        "vq": 0.0,
        "commit": 0.0,
        "lb": 0.0,
        "qb": 0.0,
        "div": 0.0,
        "n": 0,
    }
    usage_total = None
    for (xb,) in loader:
        xb = xb.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            out = model(xb)
            loss = out["loss"]
            if is_train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        bs = xb.size(0)
        totals["loss"] += float(out["loss"].item()) * bs
        totals["rec"] += float(out["recon_loss"].item()) * bs
        totals["vq"] += float(out["vq_loss"].item()) * bs
        totals["commit"] += float(out["commit_loss"].item()) * bs
        lb_val = out.get("load_balance_loss", None)
        if lb_val is not None:
            totals["lb"] += float(lb_val.item()) * bs
        qb_val = out.get("quant_balance_loss", None)
        if qb_val is not None:
            totals["qb"] += float(qb_val.item()) * bs
        div_val = out.get("diversity_loss", None)
        if div_val is not None:
            totals["div"] += float(div_val.item()) * bs
        totals["n"] += bs
        usage_hits = out.get("usage_hits", None)
        if usage_hits is not None:
            if usage_total is None:
                usage_total = usage_hits.detach().clone()
            else:
                usage_total = usage_total + usage_hits.detach()

    if is_dist and dist.is_initialized():
        dev = torch.device(device)
        stats = torch.tensor(
            [
                float(totals["loss"]),
                float(totals["rec"]),
                float(totals["vq"]),
                float(totals["commit"]),
                float(totals["lb"]),
                float(totals["qb"]),
                float(totals["div"]),
                float(totals["n"]),
            ],
            device=dev,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        totals["loss"] = float(stats[0].item())
        totals["rec"] = float(stats[1].item())
        totals["vq"] = float(stats[2].item())
        totals["commit"] = float(stats[3].item())
        totals["lb"] = float(stats[4].item())
        totals["qb"] = float(stats[5].item())
        totals["div"] = float(stats[6].item())
        totals["n"] = int(round(float(stats[7].item())))
        if usage_total is not None:
            dist.all_reduce(usage_total, op=dist.ReduceOp.SUM)

    n = max(1, totals["n"])
    if usage_total is None:
        active_codes = 0
        total_codes = 0
        utilization = 0.0
    else:
        layer_caps = getattr(core_model, "layer_codebook_sizes", None)
        num_splits = int(getattr(core_model, "num_splits", usage_total.size(1)))
        if layer_caps is not None and usage_total.dim() == 3:
            caps = torch.as_tensor(layer_caps, device=usage_total.device, dtype=torch.long).view(-1)
            if int(caps.numel()) == int(usage_total.size(0)) and int(usage_total.size(1)) == num_splits:
                active_codes = 0
                for b in range(int(caps.numel())):
                    cap = int(caps[b].item())
                    if cap <= 0:
                        continue
                    active_codes += int((usage_total[b, :, :cap] > 0).sum().item())
                total_codes = int(num_splits * int(caps.sum().item()))
            else:
                active_codes = int((usage_total > 0).sum().item())
                total_codes = int(usage_total.numel())
        else:
            active_codes = int((usage_total > 0).sum().item())
            total_codes = int(usage_total.numel())
        utilization = float(active_codes / max(1, total_codes))
    return {
        "loss": totals["loss"] / n,
        "rec": totals["rec"] / n,
        "vq": totals["vq"] / n,
        "commit": totals["commit"] / n,
        "lb": totals["lb"] / n,
        "qb": totals["qb"] / n,
        "div": totals["div"] / n,
        "active_codes": active_codes,
        "total_codes": total_codes,
        "utilization": utilization,
    }


@torch.no_grad()
def opq_alt_update_rotations(
    model,
    loader,
    device: str = "cuda",
    is_dist: bool = False,
):
    device_obj = torch.device(device)
    core_model = model.module if hasattr(model, "module") else model
    quant = core_model.quantizer
    if quant.rot_mats is None:
        raise RuntimeError("rot_mats is not initialized")

    was_training = core_model.training
    core_model.eval()

    latent_dim = int(quant.latent_dim)
    num_layers = int(quant.num_layers)
    covs = [
        torch.zeros(latent_dim, latent_dim, device=device_obj, dtype=torch.float64)
        for _ in range(num_layers)
    ]
    total_items = 0

    for (xb,) in loader:
        xb = xb.to(device_obj, non_blocking=True)
        z = core_model.encoder(xb)
        residual = z
        total_items += int(xb.size(0))

        for b in range(num_layers):
            active_v = int(quant.layer_codebook_sizes[b].item())
            rot_b = quant.rot_mats[b].to(device=z.device, dtype=z.dtype)
            residual_rot = residual @ rot_b

            residual_chunks = residual_rot.view(z.size(0), quant.num_splits, quant.split_dim)
            q_chunks = []
            for k in range(int(quant.num_splits)):
                xk = residual_chunks[:, k, :]
                cb = quant.codebooks[b, k]
                dist_mat = quant._pairwise_l2(xk, cb[:active_v])
                idx = dist_mat.argmin(dim=-1)
                q = F.embedding(idx, cb[:active_v])
                q_chunks.append(q)

            layer_q_rot = torch.stack(q_chunks, dim=1).reshape(z.size(0), latent_dim)
            covs[b] += residual.transpose(0, 1).double() @ layer_q_rot.double()
            layer_q = layer_q_rot @ rot_b.t()
            residual = residual - layer_q

    if is_dist and dist.is_initialized():
        for cov in covs:
            dist.all_reduce(cov, op=dist.ReduceOp.SUM)

    for b in range(num_layers):
        m = covs[b].cpu()
        u, _, vh = torch.linalg.svd(m, full_matrices=False)
        rot = (u @ vh).to(device=quant.rot_mats.device, dtype=quant.rot_mats.dtype)
        quant.rot_mats.data[b].copy_(rot)

    if is_dist and dist.is_initialized():
        dist.barrier()

    if was_training:
        core_model.train()

    return {"num_items": int(total_items)}


def build_slot_sizes_from_layer_caps(
    num_layers: int,
    num_splits: int,
    codebook_size: int,
    layer_codebook_sizes: tuple[int, ...] | None = None,
) -> np.ndarray:
    if layer_codebook_sizes is None:
        caps = [int(codebook_size)] * int(num_layers)
    else:
        caps = [int(v) for v in layer_codebook_sizes]
        if len(caps) != int(num_layers):
            raise ValueError(
                f"layer_codebook_sizes length={len(caps)} must equal num_layers={num_layers}"
            )
        if any(v <= 0 or v > int(codebook_size) for v in caps):
            raise ValueError(
                f"each layer_codebook_sizes entry must be in [1,{codebook_size}], got {caps}"
            )
    slot_sizes = np.array([caps[b] for b in range(int(num_layers)) for _ in range(int(num_splits))], dtype=np.int64)
    return slot_sizes


def local_to_global_ids(
    local_ids: np.ndarray,
    codebook_size: int,
    layer_codebook_sizes: tuple[int, ...] | None = None,
) -> np.ndarray:
    if local_ids.ndim != 3:
        raise ValueError(f"local_ids must be [N,B,K], got {local_ids.shape}")
    _, b_layers, k_splits = local_ids.shape
    slot_sizes = build_slot_sizes_from_layer_caps(
        num_layers=int(b_layers),
        num_splits=int(k_splits),
        codebook_size=int(codebook_size),
        layer_codebook_sizes=layer_codebook_sizes,
    )
    offsets_flat = np.zeros_like(slot_sizes, dtype=np.int64)
    if int(slot_sizes.size) > 1:
        offsets_flat[1:] = np.cumsum(slot_sizes[:-1], dtype=np.int64)
    offsets = offsets_flat.reshape(b_layers, k_splits)
    flat_local = local_ids.astype(np.int64, copy=False).reshape(local_ids.shape[0], -1)
    bad = (flat_local < 0) | (flat_local >= slot_sizes[None, :])
    if bool(np.any(bad)):
        first = np.argwhere(bad)[0]
        i = int(first[0])
        s = int(first[1])
        raise ValueError(
            f"local_ids out of range at item={i}, slot={s}: "
            f"id={int(flat_local[i, s])}, slot_size={int(slot_sizes[s])}"
        )
    return local_ids.astype(np.int64, copy=False) + offsets[None, :, :]


@lru_cache(maxsize=None)
def _l1_signed_offsets(num_dims: int, radius: int):
    if num_dims <= 0 or radius <= 0:
        return tuple()

    abs_parts = []

    def split_abs(idx: int, remain: int, cur):
        if idx == num_dims - 1:
            abs_parts.append(tuple(cur + [remain]))
            return
        for v in range(remain + 1):
            split_abs(idx + 1, remain - v, cur + [v])

    split_abs(0, int(radius), [])

    out = []
    for part in abs_parts:
        nonzero = [i for i, v in enumerate(part) if v > 0]
        if not nonzero:
            continue
        signs = [1] * len(nonzero)
        while True:
            vec = [0] * num_dims
            for j, dim in enumerate(nonzero):
                vec[dim] = signs[j] * part[dim]
            out.append(tuple(vec))

            carry = False
            for j in range(len(signs)):
                if signs[j] == 1:
                    signs[j] = -1
                    carry = True
                    break
                signs[j] = 1
            if not carry:
                break
    return tuple(out)


def _increment_mixed_radix(vec: np.ndarray, dim_order: list[int], bases: np.ndarray) -> bool:
    for d in dim_order:
        vec[d] += 1
        if vec[d] < int(bases[d]):
            return True
        vec[d] = 0
    return False


def _search_l1_nearest_in_dims(
    base_vec: np.ndarray,
    used: set[tuple[int, ...]],
    dims: tuple[int, ...],
    dim_caps: np.ndarray,
):
    if len(dims) == 0:
        return None, {"method": "failed"}

    max_radius = int(sum(max(0, int(dim_caps[d]) - 1) for d in dims))
    for radius in range(1, max_radius + 1):
        for offset in _l1_signed_offsets(len(dims), radius):
            cand = base_vec.copy()
            ok = True
            for j, d in enumerate(dims):
                v = int(cand[d]) + int(offset[j])
                if v < 0 or v >= int(dim_caps[d]):
                    ok = False
                    break
                cand[d] = v
            if not ok:
                continue
            key = tuple(int(x) for x in cand.tolist())
            if key not in used:
                return cand, {"method": "l1", "dims": len(dims), "radius": radius}
    return None, {"method": "failed"}


def _find_available_sid_last_then_all(
    base_vec: np.ndarray,
    used: set[tuple[int, ...]],
    num_layers: int,
    num_splits: int,
    dim_caps: np.ndarray,
):
    total_dims = int(num_layers * num_splits)
    last_dims = tuple(range(total_dims - num_splits, total_dims))
    cand, info = _search_l1_nearest_in_dims(
        base_vec=base_vec,
        used=used,
        dims=last_dims,
        dim_caps=dim_caps,
    )
    if cand is not None:
        return cand, info

    dim_order = list(range(total_dims - 1, -1, -1))
    cand = base_vec.copy()
    step = 0
    while _increment_mixed_radix(cand, dim_order=dim_order, bases=dim_caps):
        step += 1
        key = tuple(int(x) for x in cand.tolist())
        if key not in used:
            return cand, {"method": "odometer_full", "dims": total_dims, "steps": step}
    return None, {"method": "failed"}


def deduplicate_sid_local_ids(
    local_ids: np.ndarray,
    num_layers: int,
    num_splits: int,
    codebook_size: int,
    layer_codebook_sizes: tuple[int, ...] | None = None,
    progress_every: int = 5000,
):
    t0 = time.time()
    if local_ids.ndim != 3:
        raise ValueError(f"local_ids must be [N,B,K], got {local_ids.shape}")
    n, b_layers, k_splits = local_ids.shape
    if b_layers != num_layers or k_splits != num_splits:
        raise ValueError(
            f"SID shape mismatch: local_ids has {(b_layers, k_splits)}, expected {(num_layers, num_splits)}"
        )

    flat = local_ids.reshape(n, num_layers * num_splits).astype(np.int64, copy=True)
    t_group = time.time()
    groups: dict[tuple[int, ...], list[int]] = {}
    for i in range(n):
        key = tuple(int(x) for x in flat[i].tolist())
        groups.setdefault(key, []).append(i)

    unique_before = int(len(groups))
    collision_groups = int(sum(1 for idxs in groups.values() if len(idxs) > 1))
    collision_items = int(sum(len(idxs) - 1 for idxs in groups.values() if len(idxs) > 1))
    t_group = time.time() - t_group
    if layer_codebook_sizes is None:
        layer_caps = [int(codebook_size)] * int(num_layers)
    else:
        layer_caps = [int(v) for v in layer_codebook_sizes]
        if len(layer_caps) != int(num_layers):
            raise ValueError(
                f"layer_codebook_sizes length={len(layer_caps)} must equal num_layers={num_layers}"
            )
    dim_caps = np.asarray(
        [layer_caps[b] for b in range(num_layers) for _ in range(num_splits)],
        dtype=np.int64,
    )

    total_space = 1
    for cap in dim_caps.tolist():
        total_space *= int(cap)
    direct_space_limit = 50_000_000
    use_direct = bool(total_space >= int(n) and total_space <= int(direct_space_limit))

    changed = 0
    processed = 0
    unresolved = []
    method_count = {"l1": 0, "odometer_full": 0, "failed": 0}
    method_time_sec = {"l1": 0.0, "odometer_full": 0.0, "failed": 0.0}
    progress_every = int(progress_every)
    log_every = progress_every if progress_every > 0 else 0
    t_progress_last = time.time()
    processed_last = 0

    if use_direct:
        total_dims = int(num_layers * num_splits)
        total_space_int = int(total_space)

        t_encode = time.time()
        codes = np.zeros(n, dtype=np.int64)
        for d in range(total_dims):
            codes = codes * int(dim_caps[d]) + flat[:, d]
        encode_time = time.time() - t_encode

        strides = np.empty(total_dims, dtype=np.int64)
        stride = 1
        for d in range(total_dims - 1, -1, -1):
            strides[d] = int(stride)
            stride *= int(dim_caps[d])

        occupied = np.zeros(total_space_int, dtype=np.bool_)
        parent = np.arange(total_space_int + 1, dtype=np.int64)
        parent[total_space_int] = total_space_int

        def _find_parent(x: int) -> int:
            x = int(x)
            while True:
                px = int(parent[x])
                if px == x:
                    return x
                ppx = int(parent[px])
                if ppx != px:
                    parent[x] = ppx
                x = px

        def _occupy_code(x: int) -> None:
            x = int(x)
            if occupied[x]:
                return
            occupied[x] = True
            parent[x] = _find_parent(x + 1)

        t_occ_init = time.time()
        for c in np.unique(codes):
            _occupy_code(int(c))
        occ_init_time = time.time() - t_occ_init

        last_dims = tuple(range(total_dims - num_splits, total_dims))
        max_radius = int(sum(max(0, int(dim_caps[d]) - 1) for d in last_dims))
        l1_offsets = []
        for radius in range(1, max_radius + 1):
            for off in _l1_signed_offsets(len(last_dims), radius):
                l1_offsets.append(off)
        temp_last = np.empty(len(last_dims), dtype=np.int64)

        for _, idxs in groups.items():
            if len(idxs) <= 1:
                continue
            for dup_idx in idxs[1:]:
                base_vec = flat[dup_idx].copy()
                base_code = int(codes[dup_idx])
                t_search = time.time()
                method = "failed"

                found = False
                for off in l1_offsets:
                    delta = 0
                    ok = True
                    for j, d in enumerate(last_dims):
                        v = int(base_vec[d]) + int(off[j])
                        if v < 0 or v >= int(dim_caps[d]):
                            ok = False
                            break
                        temp_last[j] = int(v)
                        delta += int(off[j]) * int(strides[d])
                    if not ok:
                        continue
                    cand_code = int(base_code + delta)
                    if cand_code < 0 or cand_code >= total_space_int:
                        continue
                    if not bool(occupied[cand_code]):
                        for j, d in enumerate(last_dims):
                            base_vec[d] = int(temp_last[j])
                        flat[dup_idx] = base_vec
                        codes[dup_idx] = cand_code
                        _occupy_code(cand_code)
                        changed += 1
                        method = "l1"
                        found = True
                        break

                if not found:
                    cand_code = total_space_int
                    start = int(base_code + 1)
                    if start < total_space_int:
                        cand_code = _find_parent(start)
                    if cand_code < total_space_int:
                        rem = int(cand_code)
                        for d in range(total_dims - 1, -1, -1):
                            b = int(dim_caps[d])
                            flat[dup_idx, d] = rem % b
                            rem //= b
                        codes[dup_idx] = cand_code
                        _occupy_code(cand_code)
                        changed += 1
                        method = "odometer_full"
                        found = True

                if not found:
                    cand_code = _find_parent(0)
                    if cand_code < total_space_int:
                        rem = int(cand_code)
                        for d in range(total_dims - 1, -1, -1):
                            b = int(dim_caps[d])
                            flat[dup_idx, d] = rem % b
                            rem //= b
                        codes[dup_idx] = cand_code
                        _occupy_code(cand_code)
                        changed += 1
                        method = "odometer_wrap"
                        found = True

                if not found:
                    unresolved.append(int(dup_idx))

                method_count[method] = int(method_count.get(method, 0) + 1)
                method_time_sec[method] = float(method_time_sec.get(method, 0.0) + (time.time() - t_search))

                processed += 1
                if log_every > 0 and (processed % log_every == 0 or processed == collision_items):
                    t_progress_last = time.time()
                    processed_last = processed
    else:
        used = set(groups.keys())
        for _, idxs in groups.items():
            if len(idxs) <= 1:
                continue
            for dup_idx in idxs[1:]:
                base_vec = flat[dup_idx].copy()
                t_search = time.time()
                cand, info = _find_available_sid_last_then_all(
                    base_vec=base_vec,
                    used=used,
                    num_layers=num_layers,
                    num_splits=num_splits,
                    dim_caps=dim_caps,
                )
                method = str(info.get("method", "failed"))
                if method in method_count:
                    method_count[method] += 1
                else:
                    method_count[method] = 1
                method_time_sec[method] = float(method_time_sec.get(method, 0.0) + (time.time() - t_search))

                if cand is None:
                    unresolved.append(int(dup_idx))
                else:
                    new_key = tuple(int(x) for x in cand.tolist())
                    flat[dup_idx] = cand
                    used.add(new_key)
                    changed += 1

                processed += 1
                if log_every > 0 and (processed % log_every == 0 or processed == collision_items):
                    t_progress_last = time.time()
                    processed_last = processed

    unique_after = int(np.unique(flat, axis=0).shape[0])
    remaining_collisions = int(n - unique_after)
    t_total = time.time() - t0
    report = {
        "enabled": True,
        "N": int(n),
        "B": int(num_layers),
        "K": int(num_splits),
        "V": int(codebook_size),
        "layer_codebook_sizes": layer_caps,
        "unique_before": unique_before,
        "unique_after": unique_after,
        "collision_groups_before": collision_groups,
        "collision_items_before": collision_items,
        "changed_items": int(changed),
        "remaining_collisions": remaining_collisions,
        "resolved_all": bool(remaining_collisions == 0),
        "method_count": method_count,
        "method_time_sec": {k: float(v) for k, v in method_time_sec.items()},
        "unresolved_indices_head": unresolved[:20],
        "search_policy": (
            "last_layer_l1_nearest_then_all_layer_forward_odometer_then_wrap"
            if use_direct
            else "last_layer_l1_nearest_then_all_layer_exhaustive_odometer"
        ),
        "dedup_impl": "direct_occupancy_union_find" if use_direct else "legacy_tuple_scan",
        "total_space": int(total_space),
        "direct_space_limit": int(direct_space_limit),
        "group_build_time_sec": float(t_group),
        "time_sec": float(t_total),
        "progress_every": int(log_every),
    }
    new_local = flat.reshape(local_ids.shape).astype(np.int64, copy=False)
    return new_local, report


def main():
    args = parse_args()
    is_dist, rank, world_size, local_rank, device = init_distributed(args.device)
    is_main = rank == 0

    def log(msg: str):
        if is_main:
            print(msg, flush=True)

    normalize = True
    if args.normalize is not None:
        normalize = bool(args.normalize)
    else:
        legacy_no = bool(args.no_normalize_input or args.no_normalize_recon_target)
        legacy_yes = bool(args.normalize_input or args.normalize_recon_target)
        if legacy_no:
            normalize = False
        elif legacy_yes:
            normalize = True

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    if str(args.decoder_hidden_dims).strip() == "":
        decoder_hidden_dims = None
    else:
        decoder_hidden_dims = parse_hidden_dims(args.decoder_hidden_dims)
    if args.hidden_dim is not None:
        hidden_dims = () if int(args.hidden_dim) <= 0 else (int(args.hidden_dim),)
        if decoder_hidden_dims is None:
            decoder_hidden_dims = hidden_dims
    layer_codebook_sizes = parse_layer_codebook_sizes(
        text=args.layer_codebook_sizes,
        num_layers=args.num_layers,
        codebook_size=args.codebook_size,
    )
    if args.load_balance_weight < 0.0:
        raise ValueError(f"load_balance_weight must be >=0, got {args.load_balance_weight}")
    if args.quant_balance_weight < 0.0:
        raise ValueError(f"quant_balance_weight must be >=0, got {args.quant_balance_weight}")
    if args.diversity_weight < 0.0:
        raise ValueError(f"diversity_weight must be >=0, got {args.diversity_weight}")
    if args.load_balance_tau <= 0.0:
        raise ValueError(f"load_balance_tau must be >0, got {args.load_balance_tau}")
    if args.load_balance_eps <= 0.0:
        raise ValueError(f"load_balance_eps must be >0, got {args.load_balance_eps}")
    if args.diversity_eps <= 0.0:
        raise ValueError(f"diversity_eps must be >0, got {args.diversity_eps}")
    if not (0.0 <= args.diversity_tau_init <= 1.0):
        raise ValueError(f"diversity_tau_init must be in [0,1], got {args.diversity_tau_init}")
    if not (0.0 <= args.diversity_tau_target <= 1.0):
        raise ValueError(f"diversity_tau_target must be in [0,1], got {args.diversity_tau_target}")

    embeddings_npy, asin2row_json, out_dir = resolve_paths(args)
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    if is_dist and dist.is_initialized():
        dist.barrier()
    set_seed(args.seed)

    t_data = time.time()
    emb = np.load(embeddings_npy)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {emb.shape}")
    n_items, input_dim = emb.shape
    if normalize:
        emb = l2_normalize_np(emb)
    asins = load_asins_by_row(asin2row_json, n_items)
    t_data = time.time() - t_data
    idx = np.arange(n_items)
    np.random.shuffle(idx)
    n_val = max(1, int(n_items * args.val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    x_train = torch.from_numpy(emb[train_idx]).float()
    x_val = torch.from_numpy(emb[val_idx]).float()
    train_dataset = TensorDataset(x_train)
    val_dataset = TensorDataset(x_val)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        if is_dist
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if is_dist
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    cfg = RPQConfig(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_splits=args.num_splits,
        codebook_size=args.codebook_size,
        layer_codebook_sizes=layer_codebook_sizes,
        hidden_dims=hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        dropout=args.dropout,
        beta_commit=args.beta_commit,
        load_balance_loss=bool(args.load_balance_loss),
        load_balance_weight=float(args.load_balance_weight),
        load_balance_tau=float(args.load_balance_tau),
        load_balance_eps=float(args.load_balance_eps),
        quant_balance_loss=bool(args.quant_balance_loss),
        quant_balance_weight=float(args.quant_balance_weight),
        diversity_loss=bool(args.diversity_loss),
        diversity_weight=float(args.diversity_weight),
        diversity_tau_init=float(args.diversity_tau_init),
        diversity_tau_target=float(args.diversity_tau_target),
        diversity_eps=float(args.diversity_eps),
        normalized_target=normalize,
        use_ema=True,
        ema_decay=args.ema_decay,
        ema_eps=args.ema_eps,
        dead_restart_threshold=args.dead_restart_threshold,
        dead_restart_noise=args.dead_restart_noise,
    )
    raw_model = RPQTokenizer(cfg).to(device)
    if raw_model.quantizer.rot_mats is not None:
        raw_model.quantizer.rot_mats.requires_grad_(False)
    model = DDP(
        raw_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    ) if is_dist else raw_model

    optim_params = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("quantizer.rot_mats"):
            continue
        optim_params.append(param)
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    total_steps = max(1, args.max_epochs * len(train_loader))
    warmup_steps = int(total_steps * max(0.0, args.warmup_ratio))
    if args.scheduler == "cosine":
        min_lr_ratio = min(1.0, max(0.0, float(args.min_lr_ratio)))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    ckpt_path = os.path.join(out_dir, "model.pt")

    log(
        f"[rpq-vae] dataset={args.dataset} items={n_items} "
        f"B={cfg.num_layers} K={cfg.num_splits} V={cfg.codebook_size} latent_dim={cfg.latent_dim}",
    )

    for epoch in range(1, args.max_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        tr = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=args.grad_clip,
            device=str(device),
            is_dist=is_dist,
        )
        opq_alt_update_rotations(
            model,
            train_loader,
            device=str(device),
            is_dist=is_dist,
        )
        va = run_epoch(
            model,
            val_loader,
            optimizer=None,
            scheduler=None,
            grad_clip=args.grad_clip,
            device=str(device),
            is_dist=is_dist,
        )
        cur_lr = float(optimizer.param_groups[0]["lr"])
        log(
            f"[ep {epoch:03d}] "
            f"tr_loss={tr['loss']:.5f} tr_rec={tr['rec']:.5f} "
            f"va_loss={va['loss']:.5f} va_rec={va['rec']:.5f} "
            f"lr={cur_lr:.6g}"
        )
        if va["rec"] < best_val - 1e-6:
            best_val = va["rec"]
            best_epoch = epoch
            bad_epochs = 0
            if is_main:
                torch.save(
                    {
                        "cfg": asdict(cfg),
                        "model_state": raw_model.state_dict(),
                        "best_epoch": int(best_epoch),
                        "best_val_rec": float(best_val),
                    },
                    ckpt_path,
                )
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            log(
                f"[stop] early stop at epoch={epoch}, best_epoch={best_epoch}, best_val_rec={best_val:.6f}",
            )
            break
    if is_dist and dist.is_initialized():
        dist.barrier()

    if not is_main:
        if is_dist and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Best checkpoint not found at {ckpt_path}. Training may have failed before saving."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_model.load_state_dict(ckpt["model_state"])
    raw_model.to(device).eval()

    x_all = torch.from_numpy(emb).float()
    local_ids_raw, _ = quantize_all(
        model=raw_model,
        embeddings=x_all,
        batch_size=min(args.batch_size, 512),
        device=device,
    )
    local_ids_raw = local_ids_raw.astype(np.int64, copy=False)

    local_ids_uni = local_ids_raw.copy()
    if args.dedup_sid:
        local_ids_uni, _ = deduplicate_sid_local_ids(
            local_ids=local_ids_uni,
            num_layers=cfg.num_layers,
            num_splits=cfg.num_splits,
            codebook_size=cfg.codebook_size,
            layer_codebook_sizes=cfg.layer_codebook_sizes,
            progress_every=5000,
        )

    global_ids_uni = local_to_global_ids(
        local_ids=local_ids_uni,
        codebook_size=cfg.codebook_size,
        layer_codebook_sizes=cfg.layer_codebook_sizes,
    )

    asin2sid_uni = {asins[i]: global_ids_uni[i].tolist() for i in range(n_items)}
    save_json(os.path.join(out_dir, "asin2sid_uni.json"), asin2sid_uni)

    slot_sizes_flat = build_slot_sizes_from_layer_caps(
        num_layers=cfg.num_layers,
        num_splits=cfg.num_splits,
        codebook_size=cfg.codebook_size,
        layer_codebook_sizes=cfg.layer_codebook_sizes,
    )
    slot_offsets_flat = np.zeros_like(slot_sizes_flat, dtype=np.int64)
    if int(slot_sizes_flat.size) > 1:
        slot_offsets_flat[1:] = np.cumsum(slot_sizes_flat[:-1], dtype=np.int64)

    layer_caps = cfg.layer_codebook_sizes
    if layer_caps is None:
        layer_caps = [int(v) for v in raw_model.quantizer.layer_codebook_sizes.detach().cpu().tolist()]

    meta = {
        "B": int(cfg.num_layers),
        "K": int(cfg.num_splits),
        "layer_codebook_sizes": [int(v) for v in layer_caps],
        "V_global": int(cfg.vocab_global),
        "slot_codebook_sizes": slot_sizes_flat.tolist(),
        "slot_offsets": slot_offsets_flat.tolist(),
        "best_epoch": int(ckpt.get("best_epoch", -1)),
        "best_val_rec": float(ckpt.get("best_val_rec", 0.0)),
    }
    save_json(os.path.join(out_dir, "meta.json"), meta)
    print(f"[done] exported Grid-SID to {out_dir}", flush=True)

    if is_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
