#!/usr/bin/env python3
"""Train Semantic Block Diffusion on Grid-SID sequences."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import (
    SBDCollator,
    SBDDataset,
    load_json,
    load_sid_meta_or_infer,
    resolve_default_paths,
)
from model import (
    SBDModel,
    clean_alignment_cross_entropy_weighted,
    masked_cross_entropy_weighted,
    sample_timestep_mask_ratio,
)
from tuple_trie_decode import build_tuple_trie, eval_recall_ndcg_next_item_trie
from tuple_trie_decode import build_catalog_tokens


FIXED_DDP_STATIC_GRAPH = True
FIXED_DDP_GRADIENT_AS_BUCKET_VIEW = True
FIXED_DDP_BUCKET_CAP_MB = 50.0
FIXED_TF32 = True
FIXED_FUSED_ADAMW = True
FIXED_PERSISTENT_WORKERS = True
FIXED_PREFETCH_FACTOR = 4
FIXED_BUCKET_BY_LENGTH = True
FIXED_BUCKET_SIZE_MULTIPLIER = 50


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return False, rank, world_size, local_rank
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed mode requires CUDA.")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def reduce_eval_metrics(
    local_metrics: Dict[str, float],
    topk_list: List[int],
    device: torch.device,
    is_dist: bool,
) -> Dict[str, float]:
    if not is_dist:
        return local_metrics
    cnt = float(local_metrics.get("count", 0.0))
    vec = [cnt]
    for k in topk_list:
        r = float(local_metrics.get(f"recall@{k}", 0.0))
        n = float(local_metrics.get(f"ndcg@{k}", 0.0))
        vec.append(r * cnt)
        vec.append(n * cnt)
    t = torch.tensor(vec, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    out: Dict[str, float] = {}
    cnt_all = float(t[0].item())
    idx = 1
    for k in topk_list:
        hit_all = float(t[idx].item())
        ndcg_all = float(t[idx + 1].item())
        idx += 2
        out[f"recall@{k}"] = hit_all / max(1.0, cnt_all)
        out[f"ndcg@{k}"] = ndcg_all / max(1.0, cnt_all)
    out["count"] = cnt_all
    return out


def reorder_indices_by_length(
    indices: List[int],
    lengths: List[int],
    batch_size: int,
    bucket_size_multiplier: int,
    shuffle: bool,
    seed: int,
) -> List[int]:
    if len(indices) <= 1 or batch_size <= 1:
        return indices
    if bucket_size_multiplier <= 1:
        return indices

    bucket_size = int(max(batch_size, batch_size * bucket_size_multiplier))
    if bucket_size <= batch_size:
        return indices

    g = torch.Generator()
    g.manual_seed(int(seed))

    out: List[int] = []
    for start in range(0, len(indices), bucket_size):
        chunk = list(indices[start : start + bucket_size])
        chunk.sort(key=lambda idx: int(lengths[idx]), reverse=True)
        if shuffle and len(chunk) > batch_size:
            batches = [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]
            perm = torch.randperm(len(batches), generator=g).tolist()
            chunk = [idx for batch_idx in perm for idx in batches[batch_idx]]
        out.extend(chunk)
    return out


class LengthBucketSampler(Sampler[int]):
    def __init__(
        self,
        data_source,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        bucket_size_multiplier: int = 50,
    ):
        super().__init__()
        self.data_source = data_source
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_size_multiplier = int(bucket_size_multiplier)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.data_source), generator=g).tolist()
        else:
            indices = list(range(len(self.data_source)))
        indices = reorder_indices_by_length(
            indices=indices,
            lengths=self.lengths,
            batch_size=self.batch_size,
            bucket_size_multiplier=self.bucket_size_multiplier,
            shuffle=self.shuffle,
            seed=self.seed + self.epoch,
        )
        return iter(indices)

    def __len__(self) -> int:
        return len(self.data_source)


class LengthBucketDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        lengths: List[int],
        batch_size: int,
        bucket_size_multiplier: int = 50,
        **kwargs,
    ):
        super().__init__(dataset, **kwargs)
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.bucket_size_multiplier = int(bucket_size_multiplier)

    def __iter__(self):
        indices = list(super().__iter__())
        indices = reorder_indices_by_length(
            indices=indices,
            lengths=self.lengths,
            batch_size=self.batch_size,
            bucket_size_multiplier=self.bucket_size_multiplier,
            shuffle=bool(self.shuffle),
            seed=int(self.seed + self.epoch + self.rank * 9973),
        )
        return iter(indices)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="Video_Games")

    p.add_argument("--train_split", type=str, default="")
    p.add_argument("--val_split", type=str, default="")
    p.add_argument("--test_split", type=str, default="")
    p.add_argument("--asin2sid", type=str, default="")
    p.add_argument("--sid_meta", type=str, default="")
    p.add_argument(
        "--max_seq_len",
        type=int,
        default=50,
        help="Tail truncation length for each user sequence. <=0 disables truncation.",
    )

    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument(
        "--ffn_dim",
        type=int,
        default=1024,
        help="FFN intermediate dim. <=0 means using 4*hidden_dim.",
    )
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mask_prob", type=float, default=-1.0)
    p.add_argument("--diffusion_steps", type=int, default=1000)
    p.add_argument("--mask_ratio_min", type=float, default=0.0)
    p.add_argument("--mask_ratio_max", type=float, default=1.0)
    p.add_argument("--full_mask_ratio", type=float, default=0.2)
    p.add_argument(
        "--seq_ratio_gamma",
        type=float,
        default=1.0,
        help="Strength of inverse-mask-ratio weighting. 0 disables ratio effect.",
    )
    p.add_argument("--seq_ratio_eps", type=float, default=0.05, help="Numerical floor for mask ratio in weighting.")
    p.add_argument("--seq_ratio_clip_min", type=float, default=0.5, help="Minimum clip for seq ratio weight.")
    p.add_argument("--seq_ratio_clip_max", type=float, default=1.5, help="Maximum clip for seq ratio weight.")
    p.add_argument("--seq_ratio_normalize", dest="seq_ratio_normalize", action="store_true")
    p.add_argument("--no_seq_ratio_normalize", dest="seq_ratio_normalize", action="store_false")
    p.add_argument("--loss_w_seq_causal", type=float, default=1.0)
    p.add_argument(
        "--loss_w_clean_align",
        type=float,
        default=0.25,
        help="Weight for clean-layer alignment CE on the causal branch.",
    )
    p.add_argument(
        "--loss_w_bidir_aux",
        type=float,
        default=0.2,
        help="Weight for the auxiliary bidirectional branch loss.",
    )
    p.add_argument(
        "--loss_w_bidir_clean_align",
        type=float,
        default=0.25,
        help="Inner weight for clean-layer alignment CE on the bidirectional branch.",
    )
    p.set_defaults(seq_ratio_normalize=True)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=2e-4)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)

    p.add_argument("--early_stop_patience", type=int, default=6)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--beam_size", type=int, default=50)
    p.add_argument(
        "--decode_fill_plan",
        type=str,
        default="auto",
        help='Comma-separated positive integers (e.g. "1,2,1") or "auto" (= [1] * K).',
    )
    p.add_argument("--topk_list", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument(
        "--best_metric_k",
        type=int,
        default=10,
        help="Select best ckpt by val recall@k (must be in --topk_list; fallback to largest k).",
    )
    p.add_argument(
        "--val_eval_ratio",
        type=float,
        default=0.1,
        help="Use only this ratio of validation users for periodic val evaluation (0,1].",
    )
    p.add_argument(
        "--test_eval_ratio",
        type=float,
        default=1.0,
        help="Use only this ratio of test users for final test evaluation (0,1].",
    )
    p.add_argument("--eval_max_users", type=int, default=0)
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--eval_cache_batch_size", type=int, default=64)

    p.add_argument(
        "--train_sdp_backend",
        type=str,
        default="auto",
        choices=["auto", "math", "flash", "mem_efficient"],
        help="Training scaled-dot-product attention backend.",
    )
    p.add_argument(
        "--eval_sdp_backend",
        type=str,
        default="auto",
        choices=["auto", "math", "flash", "mem_efficient"],
        help="Evaluation/generation scaled-dot-product attention backend.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--log_path", type=str, default="")
    p.add_argument("--ckpt_name", type=str, default="")
    p.add_argument("--resume", type=int, default=0, choices=[0, 1])
    p.add_argument("--resume_path", type=str, default="")
    return p.parse_args()


def resolve_paths(args):
    defaults = resolve_default_paths(dataset=args.dataset)
    return {
        "train_split": args.train_split or defaults["train_split"],
        "val_split": args.val_split or defaults["val_split"],
        "test_split": args.test_split or defaults["test_split"],
        "asin2sid": args.asin2sid or defaults["sid_json"],
        "sid_meta": args.sid_meta or defaults["sid_meta"],
    }


def parse_fill_plan(raw: str) -> List[int]:
    text = str(raw).strip()
    if text == "":
        return []
    parts = [x.strip() for x in text.split(",") if x.strip() != ""]
    out = [int(x) for x in parts]
    if len(out) == 0:
        return []
    if any(v <= 0 for v in out):
        raise ValueError("--decode_fill_plan entries must be > 0")
    return out


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float, min_lr_ratio: float):
    warmup_steps = int(total_steps * max(0.0, warmup_ratio))
    min_lr_ratio = min(1.0, max(0.0, float(min_lr_ratio)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def format_metrics(metrics: Dict[str, float]) -> str:
    info = [f"{k}={v:.8f}" for k, v in metrics.items() if k != "count"]
    info.append(f"cnt={int(metrics.get('count', 0))}")
    return " ".join(info)


def select_eval_sequences(
    sequences: List[List[str]],
    ratio: float,
    max_users: int,
) -> List[List[str]]:
    if not (0.0 < float(ratio) <= 1.0):
        raise ValueError(f"eval ratio must be in (0,1], got {ratio}")
    n_total = len(sequences)
    if n_total == 0:
        return sequences
    n_keep = int(math.ceil(n_total * float(ratio)))
    n_keep = max(1, min(n_total, n_keep))
    out = sequences[:n_keep]
    if int(max_users) > 0:
        out = out[: int(max_users)]
    return out


def load_state_dict_compat(model: SBDModel, state_dict: Dict[str, torch.Tensor]) -> None:
    legacy_drop_keys = [
        "layer_aggregator.weight",
        "layer_aggregator.bias",
        "layer_agg_weight",
        "layer_agg_bias",
        "module.layer_aggregator.weight",
        "module.layer_aggregator.bias",
        "module.layer_agg_weight",
        "module.layer_agg_bias",
    ]
    for key in legacy_drop_keys:
        state_dict.pop(key, None)

    state_dict.pop("item_id_embed.weight", None)
    state_dict.pop("clean_id_marker_pe", None)
    state_dict.pop("module.item_id_embed.weight", None)
    state_dict.pop("module.clean_id_marker_pe", None)

    model.load_state_dict(state_dict, strict=True)


def sample_layerwise_next_task_noise(
    clean_sid_ids: torch.Tensor,
    item_mask: torch.Tensor,
    mask_ratio: torch.Tensor,
    layer_full_mask_ratio: float,
    mask_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if clean_sid_ids.dim() != 4:
        raise ValueError(f"clean_sid_ids must be [B,n,B,K], got {tuple(clean_sid_ids.shape)}")
    if item_mask.dim() != 2:
        raise ValueError(f"item_mask must be [B,n], got {tuple(item_mask.shape)}")
    bsz, n, b_rq, k_pq = clean_sid_ids.shape
    if item_mask.shape != (bsz, n):
        raise ValueError("item_mask shape mismatch")
    if mask_ratio.shape != (bsz, n, b_rq):
        raise ValueError(
            f"mask_ratio tensor must be [B,n,B_rq], got {tuple(mask_ratio.shape)} expected {(bsz, n, b_rq)}"
        )
    if not (0.0 <= float(layer_full_mask_ratio) <= 1.0):
        raise ValueError(f"layer_full_mask_ratio must be in [0,1], got {layer_full_mask_ratio}")

    dev = clean_sid_ids.device
    valid_item = item_mask.to(torch.bool)
    valid_layer = valid_item[:, :, None].expand(-1, -1, b_rq)
    valid = valid_layer[:, :, :, None].expand(-1, -1, -1, k_pq)

    ratio = mask_ratio.to(device=dev, dtype=torch.float32).clamp(0.0, 1.0)
    rand_mask = (
        torch.rand((bsz, n, b_rq, k_pq), device=dev, dtype=torch.float32) < ratio[:, :, :, None]
    ) & valid

    full_layer = torch.zeros((bsz, n, b_rq), dtype=torch.bool, device=dev)
    if layer_full_mask_ratio > 0.0:
        valid_layer_flat = valid_layer.reshape(bsz, n * b_rq)
        full_layer_flat = full_layer.reshape(bsz, n * b_rq)
        for i in range(bsz):
            valid_idx = torch.where(valid_layer_flat[i])[0]
            li = int(valid_idx.numel())
            if li <= 0:
                continue
            cnt = int(round(li * float(layer_full_mask_ratio)))
            cnt = max(0, min(li, cnt))
            if cnt == 0:
                continue
            perm = torch.randperm(li, device=dev, dtype=torch.long)
            pick = valid_idx[perm[:cnt]]
            full_layer_flat[i, pick] = True

    full_mask = full_layer[:, :, :, None].expand(-1, -1, -1, k_pq)
    loss_mask = (rand_mask | full_mask) & valid

    noisy = clean_sid_ids.clone()
    noisy[loss_mask] = int(mask_value)
    return noisy, loss_mask


def build_seq_ratio_token_weights(
    loss_mask: torch.Tensor,
    item_mask: torch.Tensor,
    granularity: str,
    gamma: float,
    eps: float,
    clip_min: float,
    clip_max: float,
    normalize: bool,
) -> torch.Tensor:
    if loss_mask.dim() != 4:
        raise ValueError(f"loss_mask must be [B,n,B,K], got {tuple(loss_mask.shape)}")
    if item_mask.dim() != 2:
        raise ValueError(f"item_mask must be [B,n], got {tuple(item_mask.shape)}")
    if granularity not in {"item", "layer"}:
        raise ValueError(f"Unknown granularity={granularity}, expected item/layer")
    if gamma < 0.0:
        raise ValueError("gamma must be >= 0")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    bsz, n, b_rq, k_pq = loss_mask.shape
    if item_mask.shape != (bsz, n):
        raise ValueError("item_mask shape mismatch")

    dev = loss_mask.device
    lm = loss_mask.to(torch.float32)
    valid_item = item_mask.to(torch.bool)

    def _apply_weighting(mask_ratio: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        out = torch.ones_like(mask_ratio, dtype=torch.float32, device=dev)
        if gamma == 0.0 or (not bool(active.any())):
            return out

        ratio_active = mask_ratio[active].clamp_min(float(eps))
        ref = ratio_active.mean().clamp_min(float(eps))
        w = ((ref + float(eps)) / (ratio_active + float(eps))).pow(float(gamma))

        if clip_min > 0.0 or clip_max > 0.0:
            lo = float(clip_min) if clip_min > 0.0 else None
            hi = float(clip_max) if clip_max > 0.0 else None
            if lo is not None and hi is not None:
                w = torch.clamp(w, min=lo, max=hi)
            elif lo is not None:
                w = torch.clamp(w, min=lo)
            elif hi is not None:
                w = torch.clamp(w, max=hi)

        out[active] = w
        if normalize:
            mean_w = out[active].mean().clamp_min(1e-6)
            out[active] = out[active] / mean_w
        return out

    if granularity == "item":
        masked_cnt = lm.sum(dim=(2, 3))  # [B,n]
        ratio = torch.zeros_like(masked_cnt, dtype=torch.float32, device=dev)
        denom = float(max(1, b_rq * k_pq))
        ratio[valid_item] = masked_cnt[valid_item] / denom
        active = valid_item & (masked_cnt > 0)
        w_item = _apply_weighting(mask_ratio=ratio, active=active)
        token_w = w_item[:, :, None, None].expand(-1, -1, b_rq, k_pq)
    else:  # layer
        masked_cnt = lm.sum(dim=3)  # [B,n,B_rq]
        valid_layer = valid_item[:, :, None].expand(-1, -1, b_rq)
        ratio = torch.zeros_like(masked_cnt, dtype=torch.float32, device=dev)
        denom = float(max(1, k_pq))
        ratio[valid_layer] = masked_cnt[valid_layer] / denom
        active = valid_layer & (masked_cnt > 0)
        w_layer = _apply_weighting(mask_ratio=ratio, active=active)
        token_w = w_layer[:, :, :, None].expand(-1, -1, -1, k_pq)

    token_w = torch.where(loss_mask, token_w, torch.ones_like(token_w, device=dev))
    return token_w


def apply_ddp_global_mean_correction(loss: torch.Tensor, local_den: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return loss
    local_den = local_den.detach().to(device=loss.device, dtype=torch.float32)
    global_den = local_den.clone()
    dist.all_reduce(global_den, op=dist.ReduceOp.SUM)
    scale = (float(dist.get_world_size()) * local_den / global_den.clamp_min(1e-12)).to(dtype=loss.dtype)
    return loss * scale


def run_epoch(
    model: SBDModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    fixed_mask_prob: float,
    diffusion_steps: int,
    mask_ratio_min: float,
    mask_ratio_max: float,
    full_mask_ratio: float,
    device: str,
    grad_clip: float,
    loss_w_seq_causal: float,
    loss_w_clean_align: float,
    loss_w_bidir_aux: float,
    loss_w_bidir_clean_align: float,
    layer_codebook_sizes: List[int],
    slot_codebook_sizes: List[int],
    slot_offsets: List[int],
    seq_ratio_gamma: float = 1.0,
    seq_ratio_eps: float = 0.05,
    seq_ratio_clip_min: float = 0.5,
    seq_ratio_clip_max: float = 1.5,
    seq_ratio_normalize: bool = True,
    show_progress: bool = False,
    progress_desc: str = "train",
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_loss_causal_target = 0.0
    total_loss_causal_clean = 0.0
    total_loss_bidir_target = 0.0
    total_loss_bidir_clean = 0.0
    total_n = 0
    pbar = tqdm(
        total=len(loader),
        desc=progress_desc,
        leave=False,
        dynamic_ncols=True,
        mininterval=1.0,
    ) if show_progress else None
    try:
        for step, batch in enumerate(loader, start=1):
            sid_ids = batch["sid_ids"].to(device, non_blocking=True)
            item_mask = batch["item_mask"].to(device, non_blocking=True)

            if fixed_mask_prob >= 0.0:
                ratio_base = torch.full(
                    (sid_ids.size(0), sid_ids.size(1), sid_ids.size(2)),
                    float(fixed_mask_prob),
                    device=sid_ids.device,
                    dtype=torch.float32,
                )
            else:
                timesteps = torch.randint(
                    1,
                    int(diffusion_steps) + 1,
                    (sid_ids.size(0), sid_ids.size(1), sid_ids.size(2)),
                    device=sid_ids.device,
                )
                ratio_base = (timesteps.to(torch.float32) / float(diffusion_steps)).clamp(
                    min=float(mask_ratio_min),
                    max=float(mask_ratio_max),
                )
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(is_train):
                raw_model = model.module if hasattr(model, "module") else model
                lm_head_weight = raw_model.lm_head.weight
                need_clean_align = float(loss_w_clean_align) > 0.0
                need_bidir = float(loss_w_bidir_aux) > 0.0
                need_bidir_clean_align = need_bidir and float(loss_w_bidir_clean_align) > 0.0

                noisy_seq, loss_mask_seq = sample_layerwise_next_task_noise(
                    clean_sid_ids=sid_ids,
                    item_mask=item_mask,
                    mask_ratio=ratio_base,
                    layer_full_mask_ratio=float(full_mask_ratio),
                    mask_value=-1,
                )
                token_weights_seq = build_seq_ratio_token_weights(
                    loss_mask=loss_mask_seq,
                    item_mask=item_mask,
                    granularity="item",
                    gamma=float(seq_ratio_gamma),
                    eps=float(seq_ratio_eps),
                    clip_min=float(seq_ratio_clip_min),
                    clip_max=float(seq_ratio_clip_max),
                    normalize=bool(seq_ratio_normalize),
                )
                clean_token_weights = item_mask[:, :, None, None].expand_as(sid_ids).to(torch.float32)

                loss = torch.zeros((), device=sid_ids.device, dtype=torch.float32)
                loss_causal_target_weighted = torch.zeros((), device=sid_ids.device, dtype=torch.float32)
                loss_causal_clean_weighted = torch.zeros((), device=sid_ids.device, dtype=torch.float32)
                loss_bidir_target_weighted = torch.zeros((), device=sid_ids.device, dtype=torch.float32)
                loss_bidir_clean_weighted = torch.zeros((), device=sid_ids.device, dtype=torch.float32)

                logits_seq_causal = model(
                    sid_ids=sid_ids,
                    item_mask=item_mask,
                    noisy_target_ids=noisy_seq,
                    attn_mode="causal",
                    return_clean_hidden=need_clean_align,
                )
                clean_hidden_causal = raw_model.pop_last_clean_hidden() if need_clean_align else None

                loss_target_causal = masked_cross_entropy_weighted(
                    logits=logits_seq_causal,
                    target_ids=sid_ids,
                    loss_mask=loss_mask_seq,
                    token_weights=token_weights_seq,
                    layer_codebook_sizes=layer_codebook_sizes,
                    slot_codebook_sizes=slot_codebook_sizes,
                    slot_offsets=slot_offsets,
                )
                loss_target_causal = apply_ddp_global_mean_correction(
                    loss=loss_target_causal,
                    local_den=token_weights_seq[loss_mask_seq].sum(),
                )
                loss_causal_target_weighted = float(loss_w_seq_causal) * loss_target_causal
                causal_branch_loss = loss_target_causal

                if need_clean_align:
                    loss_clean_causal = clean_alignment_cross_entropy_weighted(
                        clean_hidden=clean_hidden_causal,
                        target_ids=sid_ids,
                        item_mask=item_mask,
                        token_weights=clean_token_weights,
                        lm_head_weight=lm_head_weight,
                        layer_codebook_sizes=layer_codebook_sizes,
                        slot_codebook_sizes=slot_codebook_sizes,
                        slot_offsets=slot_offsets,
                    )
                    loss_clean_causal = apply_ddp_global_mean_correction(
                        loss=loss_clean_causal,
                        local_den=clean_token_weights.sum(),
                    )
                    loss_causal_clean_weighted = float(loss_w_seq_causal * loss_w_clean_align) * loss_clean_causal
                    causal_branch_loss = causal_branch_loss + float(loss_w_clean_align) * loss_clean_causal

                loss = loss + float(loss_w_seq_causal) * causal_branch_loss

                if need_bidir:
                    logits_seq_bidir = model(
                        sid_ids=sid_ids,
                        item_mask=item_mask,
                        noisy_target_ids=noisy_seq,
                        attn_mode="bidir",
                        return_clean_hidden=need_bidir_clean_align,
                    )
                    clean_hidden_bidir = raw_model.pop_last_clean_hidden() if need_bidir_clean_align else None

                    loss_target_bidir = masked_cross_entropy_weighted(
                        logits=logits_seq_bidir,
                        target_ids=sid_ids,
                        loss_mask=loss_mask_seq,
                        token_weights=token_weights_seq,
                        layer_codebook_sizes=layer_codebook_sizes,
                        slot_codebook_sizes=slot_codebook_sizes,
                        slot_offsets=slot_offsets,
                    )
                    loss_target_bidir = apply_ddp_global_mean_correction(
                        loss=loss_target_bidir,
                        local_den=token_weights_seq[loss_mask_seq].sum(),
                    )
                    loss_bidir_target_weighted = float(loss_w_bidir_aux) * loss_target_bidir
                    bidir_branch_loss = loss_target_bidir

                    if need_bidir_clean_align:
                        loss_clean_bidir = clean_alignment_cross_entropy_weighted(
                            clean_hidden=clean_hidden_bidir,
                            target_ids=sid_ids,
                            item_mask=item_mask,
                            token_weights=clean_token_weights,
                            lm_head_weight=lm_head_weight,
                            layer_codebook_sizes=layer_codebook_sizes,
                            slot_codebook_sizes=slot_codebook_sizes,
                            slot_offsets=slot_offsets,
                        )
                        loss_clean_bidir = apply_ddp_global_mean_correction(
                            loss=loss_clean_bidir,
                            local_den=clean_token_weights.sum(),
                        )
                        loss_bidir_clean_weighted = float(loss_w_bidir_aux * loss_w_bidir_clean_align) * loss_clean_bidir
                        bidir_branch_loss = bidir_branch_loss + float(loss_w_bidir_clean_align) * loss_clean_bidir

                    loss = loss + float(loss_w_bidir_aux) * bidir_branch_loss

                if is_train:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

            bs = sid_ids.size(0)
            total_loss += float(loss.item()) * bs
            total_loss_causal_target += float(loss_causal_target_weighted.item()) * bs
            total_loss_causal_clean += float(loss_causal_clean_weighted.item()) * bs
            total_loss_bidir_target += float(loss_bidir_target_weighted.item()) * bs
            total_loss_bidir_clean += float(loss_bidir_clean_weighted.item()) * bs
            total_n += bs

            if pbar is not None:
                pbar.update(1)
                if step == 1 or step % 20 == 0:
                    pbar.set_postfix(loss=f"{float(loss.item()):.4f}", samples=total_n)
    finally:
        if pbar is not None:
            pbar.close()

    if dist.is_available() and dist.is_initialized():
        t = torch.tensor(
            [
                total_loss,
                total_loss_causal_target,
                total_loss_causal_clean,
                total_loss_bidir_target,
                total_loss_bidir_clean,
                float(total_n),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss = float(t[0].item())
        total_loss_causal_target = float(t[1].item())
        total_loss_causal_clean = float(t[2].item())
        total_loss_bidir_target = float(t[3].item())
        total_loss_bidir_clean = float(t[4].item())
        total_n = int(t[5].item())
    denom = max(1, total_n)
    return {
        "total": total_loss / denom,
        "causal_target": total_loss_causal_target / denom,
        "causal_clean_align": total_loss_causal_clean / denom,
        "bidir_target": total_loss_bidir_target / denom,
        "bidir_clean_align": total_loss_bidir_clean / denom,
    }


def main():
    args = parse_args()
    if not args.save_dir:
        args.save_dir = f"outputs/{args.dataset}/sbd/checkpoints"
    if not args.log_path:
        args.log_path = f"outputs/{args.dataset}/sbd/logs/sbd.log"
    if not (0.0 <= args.full_mask_ratio <= 1.0):
        raise ValueError("--full_mask_ratio must be in [0, 1]")
    if args.loss_w_seq_causal < 0.0:
        raise ValueError("--loss_w_seq_causal must be >= 0")
    if args.loss_w_clean_align < 0.0:
        raise ValueError("--loss_w_clean_align must be >= 0")
    if args.loss_w_bidir_aux < 0.0:
        raise ValueError("--loss_w_bidir_aux must be >= 0")
    if args.loss_w_bidir_clean_align < 0.0:
        raise ValueError("--loss_w_bidir_clean_align must be >= 0")
    if args.seq_ratio_gamma < 0.0:
        raise ValueError("--seq_ratio_gamma must be >= 0")
    if args.seq_ratio_eps <= 0.0:
        raise ValueError("--seq_ratio_eps must be > 0")
    if not (0.0 < float(args.val_eval_ratio) <= 1.0):
        raise ValueError("--val_eval_ratio must be in (0, 1]")
    if not (0.0 < float(args.test_eval_ratio) <= 1.0):
        raise ValueError("--test_eval_ratio must be in (0, 1]")
    if args.seq_ratio_clip_min > 0.0 and args.seq_ratio_clip_max > 0.0 and args.seq_ratio_clip_min > args.seq_ratio_clip_max:
        raise ValueError("--seq_ratio_clip_min must be <= --seq_ratio_clip_max")
    if args.loss_w_seq_causal <= 0.0:
        raise ValueError("--loss_w_seq_causal must be > 0 (main causal branch)")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    is_dist, rank, world_size, local_rank = init_distributed()
    is_main = (rank == 0)
    set_seed(args.seed)

    paths = resolve_paths(args)
    sid_shape = load_sid_meta_or_infer(paths["sid_meta"], paths["asin2sid"])
    b_layers = int(sid_shape["B"])
    k_splits = int(sid_shape["K"])
    v_global = int(sid_shape["V_global"])
    s_total = int(b_layers * k_splits)

    layer_caps_meta_raw = sid_shape.get("layer_codebook_sizes")
    if not (isinstance(layer_caps_meta_raw, list) and len(layer_caps_meta_raw) == b_layers):
        raise ValueError(
            "sid_meta must contain layer_codebook_sizes with length B for non-redundant vocab mode."
        )
    layer_caps_meta = [int(v) for v in layer_caps_meta_raw]
    if any(v <= 0 for v in layer_caps_meta):
        raise ValueError(f"Invalid layer_codebook_sizes={layer_caps_meta}; all entries must be > 0")

    slot_sizes_meta_raw = sid_shape.get("slot_codebook_sizes")
    slot_offsets_meta_raw = sid_shape.get("slot_offsets")
    if not (isinstance(slot_sizes_meta_raw, list) and len(slot_sizes_meta_raw) == s_total):
        raise ValueError(
            "sid_meta must contain slot_codebook_sizes with length B*K for non-redundant vocab mode."
        )
    if not (isinstance(slot_offsets_meta_raw, list) and len(slot_offsets_meta_raw) == s_total):
        raise ValueError(
            "sid_meta must contain slot_offsets with length B*K for non-redundant vocab mode."
        )

    slot_codebook_sizes = [int(v) for v in slot_sizes_meta_raw]
    slot_offsets = [int(v) for v in slot_offsets_meta_raw]
    if any(v <= 0 for v in slot_codebook_sizes):
        raise ValueError(f"Invalid slot_codebook_sizes={slot_codebook_sizes}")
    if any(v < 0 for v in slot_offsets):
        raise ValueError(f"Invalid slot_offsets={slot_offsets}")

    expected_offsets = [0] * s_total
    for i in range(1, s_total):
        expected_offsets[i] = int(expected_offsets[i - 1] + slot_codebook_sizes[i - 1])
    if slot_offsets != expected_offsets:
        raise ValueError(
            "Non-redundant SID requires contiguous slot_offsets. "
            f"expected={expected_offsets}, got={slot_offsets}"
        )
    if int(sum(slot_codebook_sizes)) != v_global:
        raise ValueError(
            f"Non-redundant SID requires V_global == sum(slot_codebook_sizes), "
            f"got V_global={v_global}, sum={int(sum(slot_codebook_sizes))}"
        )

    for b in range(b_layers):
        expected = int(layer_caps_meta[b])
        for k in range(k_splits):
            s = b * k_splits + k
            if int(slot_codebook_sizes[s]) != expected:
                raise ValueError(
                    f"slot_codebook_sizes mismatch at layer={b}, split={k}: "
                    f"slot={slot_codebook_sizes[s]} vs layer_cap={expected}"
                )
    layer_codebook_sizes = [int(v) for v in layer_caps_meta]
    local_vocab_layout = "slot_offsets_meta_non_redundant"

    fill_raw = str(args.decode_fill_plan).strip()
    if fill_raw.lower() == "auto":
        decode_fill_plan = [1 for _ in range(max(1, k_splits))]
    else:
        decode_fill_plan = parse_fill_plan(fill_raw)
    if not decode_fill_plan:
        raise ValueError("--decode_fill_plan cannot be empty")

    if is_dist:
        device = f"cuda:{local_rank}"
    else:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        else:
            device = args.device

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda and FIXED_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    ckpt_name = args.ckpt_name.strip() or args.dataset
    save_path = os.path.join(args.save_dir, f"{ckpt_name}.pt")
    meta_path = os.path.join(args.save_dir, f"{ckpt_name}.meta.json")
    train_state_path = os.path.join(args.save_dir, f"{ckpt_name}.train_state.pt")

    def log(msg: str):
        if not is_main:
            return
        print(msg)
        with open(args.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    train_ds = SBDDataset(
        split_path=paths["train_split"],
        asin2sid_json=paths["asin2sid"],
        strict=True,
        max_seq_len=args.max_seq_len,
    )
    val_ds = SBDDataset(
        split_path=paths["val_split"],
        asin2sid_json=paths["asin2sid"],
        strict=True,
        max_seq_len=args.max_seq_len,
    )
    test_ds = SBDDataset(
        split_path=paths["test_split"],
        asin2sid_json=paths["asin2sid"],
        strict=True,
        max_seq_len=args.max_seq_len,
    )

    collator = SBDCollator()
    train_lengths = [len(seq) for seq in train_ds.user_seqs]
    if FIXED_BUCKET_BY_LENGTH:
        if is_dist:
            train_sampler = LengthBucketDistributedSampler(
                train_ds,
                lengths=train_lengths,
                batch_size=args.batch_size,
                bucket_size_multiplier=FIXED_BUCKET_SIZE_MULTIPLIER,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
                seed=args.seed,
            )
        else:
            train_sampler = LengthBucketSampler(
                train_ds,
                lengths=train_lengths,
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed,
                bucket_size_multiplier=FIXED_BUCKET_SIZE_MULTIPLIER,
            )
    else:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        ) if is_dist else None
    train_loader_kwargs = dict(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )
    if args.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = FIXED_PERSISTENT_WORKERS
        train_loader_kwargs["prefetch_factor"] = FIXED_PREFETCH_FACTOR
    train_loader = DataLoader(**train_loader_kwargs)

    raw_model = SBDModel(
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_rq_layers=b_layers,
        num_pq_splits=k_splits,
        vocab_global=v_global,
        dropout=args.dropout,
        train_sdp_backend=args.train_sdp_backend,
        eval_sdp_backend=args.eval_sdp_backend,
    ).to(device)
    model = DDP(
        raw_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        static_graph=FIXED_DDP_STATIC_GRAPH,
        gradient_as_bucket_view=FIXED_DDP_GRADIENT_AS_BUCKET_VIEW,
        bucket_cap_mb=FIXED_DDP_BUCKET_CAP_MB,
    ) if is_dist else raw_model
    total_params = count_trainable_params(raw_model)

    optimizer_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay)
    if use_cuda and FIXED_FUSED_ADAMW:
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(raw_model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(raw_model.parameters(), **optimizer_kwargs)
    scheduler = None
    if args.scheduler == "cosine":
        total_steps = max(1, args.num_epochs * len(train_loader))
        scheduler = build_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_ratio=args.warmup_ratio,
            min_lr_ratio=args.min_lr_ratio,
        )

    log(
        f"[env] device={device} cuda={torch.cuda.is_available()} seed={args.seed} "
        f"dist={is_dist} rank={rank}/{world_size}"
    )
    log(
        f"[data] users train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}; "
        f"B={b_layers} K={k_splits} V_global={v_global} "
        f"layer_caps={layer_codebook_sizes if layer_codebook_sizes is not None else 'none'} "
        f"max_seq_len={args.max_seq_len}"
    )
    eff_ffn_dim = 4 * int(args.hidden_dim) if int(args.ffn_dim) <= 0 else int(args.ffn_dim)
    log(
        f"[model] hidden={args.hidden_dim} ffn={eff_ffn_dim} heads={args.num_heads} layers={args.num_layers}; "
        f"agg=mean batch={args.batch_size} params={total_params}"
    )
    log(
        f"[perf] sdp(train={args.train_sdp_backend},eval={args.eval_sdp_backend}) "
        f"loader(workers={args.num_workers},persistent={FIXED_PERSISTENT_WORKERS},"
        f"prefetch={FIXED_PREFETCH_FACTOR if args.num_workers > 0 else 'n/a'},"
        f"bucket_by_length={FIXED_BUCKET_BY_LENGTH},bucket_mult={FIXED_BUCKET_SIZE_MULTIPLIER}) "
        f"tf32={FIXED_TF32} fused_adamw={FIXED_FUSED_ADAMW} "
        f"ddp(static_graph={FIXED_DDP_STATIC_GRAPH},"
        f"grad_bucket_view={FIXED_DDP_GRADIENT_AS_BUCKET_VIEW},"
        f"bucket_cap_mb={FIXED_DDP_BUCKET_CAP_MB:.1f})"
    )
    log(
        f"[noise] mask_prob={args.mask_prob} steps={args.diffusion_steps} "
        f"ratio=[{args.mask_ratio_min:.3f},{args.mask_ratio_max:.3f}] full_mask_ratio={args.full_mask_ratio:.3f}"
    )
    log(f"[noise] seq_mask_mode=layer_p_then_token_bernoulli+full_layer_only full_mask_ratio={args.full_mask_ratio} force_mask=0")
    log(
        f"[loss] causal_scale={args.loss_w_seq_causal:.3f} "
        f"clean_align={args.loss_w_clean_align:.3f} "
        f"bidir_aux={args.loss_w_bidir_aux:.3f} "
        f"bidir_clean_align={args.loss_w_bidir_clean_align:.3f}"
    )
    log(
        f"[loss] seq_ratio_weighting=item gamma={args.seq_ratio_gamma:.3f} "
        f"eps={args.seq_ratio_eps:.4f} clip=[{args.seq_ratio_clip_min:.3f},{args.seq_ratio_clip_max:.3f}] "
        f"normalize={bool(args.seq_ratio_normalize)}"
    )
    log(
        f"[loss] local_vocab_ce=True layout={local_vocab_layout} "
        f"slot_sizes={slot_codebook_sizes} layer_caps={layer_codebook_sizes}"
    )
    log(
        f"[decode] backend=global_gpu beam={args.beam_size} "
        f"eval_temperature={args.eval_temperature} "
        f"eval_cache_batch_size={args.eval_cache_batch_size} "
        f"fill_plan={decode_fill_plan}"
    )
    asin2sid = load_json(paths["asin2sid"])
    trie_t0 = time.time()
    trie_root, trie_b, trie_k = build_tuple_trie(asin2sid)
    if trie_b != b_layers or trie_k != k_splits:
        raise ValueError(f"Trie SID shape {(trie_b, trie_k)} != model SID shape {(b_layers, k_splits)}")
    log(f"[trie] built done in {time.time()-trie_t0:.2f}s (warmup disabled)")
    catalog_tokens = build_catalog_tokens(
        asin2sid=asin2sid,
        slot_offsets=slot_offsets,
        device=device,
    )
    log("[decode] using global GPU catalog-state beam decoder")

    val_sequences = select_eval_sequences(
        sequences=val_ds.user_seqs,
        ratio=float(args.val_eval_ratio),
        max_users=int(args.eval_max_users),
    )
    log(
        f"[eval] val subset users={len(val_sequences)}/{len(val_ds.user_seqs)} "
        f"ratio={args.val_eval_ratio:.3f} max_users={args.eval_max_users}"
    )
    test_sequences = select_eval_sequences(
        sequences=test_ds.user_seqs,
        ratio=float(args.test_eval_ratio),
        max_users=int(args.eval_max_users),
    )
    log(
        f"[eval] test subset users={len(test_sequences)}/{len(test_ds.user_seqs)} "
        f"ratio={args.test_eval_ratio:.3f} max_users={args.eval_max_users}"
    )

    def run_eval(sequences, progress_desc: str, max_users: int = 0):
        seq = sequences[: int(max_users)] if int(max_users) > 0 else sequences
        local_sequences = seq[rank::world_size] if is_dist else seq
        eval_t0 = time.perf_counter()
        local_metrics = eval_recall_ndcg_next_item_trie(
            model=raw_model,
            sequences=local_sequences,
            asin2sid=asin2sid,
            trie_root=trie_root,
            num_layers=b_layers,
            num_splits=k_splits,
            slot_codebook_sizes=slot_codebook_sizes,
            slot_offsets=slot_offsets,
            topk_list=topk_list,
            beam_size=args.beam_size,
            decode_fill_plan=(decode_fill_plan if len(decode_fill_plan) > 0 else None),
            max_users=0,
            temperature=args.eval_temperature,
            device=device,
            cache_build_batch_size=args.eval_cache_batch_size,
            catalog_tokens=catalog_tokens,
            show_progress=(is_main and (not is_dist or rank == 0)),
            progress_desc=(f"{progress_desc}/r{rank}" if is_dist else progress_desc),
        )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        elapsed = time.perf_counter() - eval_t0
        metrics = reduce_eval_metrics(
            local_metrics,
            topk_list=topk_list,
            device=torch.device(device),
            is_dist=is_dist,
        )
        if is_dist:
            elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            elapsed = float(elapsed_tensor.item())
        if is_main:
            qps = float(metrics.get("count", 0.0)) / max(elapsed, 1e-9)
            log(f"[eval] {progress_desc} elapsed={elapsed:.3f}s qps={qps:.3f}")
        return metrics

    topk_list = sorted(set(int(x) for x in args.topk_list))
    if len(topk_list) == 0:
        raise ValueError("topk_list cannot be empty")

    best_k = int(args.best_metric_k)
    if best_k not in topk_list:
        fallback_k = int(topk_list[-1])
        log(
            f"[warn] best_metric_k={best_k} not in topk_list={topk_list}; "
            f"fallback to recall@{fallback_k}"
        )
        best_k = fallback_k
    best_key = f"recall@{best_k}"

    best_val = float("-inf")
    best_epoch = 0
    no_improve = 0
    best_val_metrics: Dict[str, float] = {}
    start_epoch = 1

    if bool(args.resume):
        resume_path = args.resume_path.strip() or train_state_path
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"resume enabled but checkpoint not found: {resume_path}")
        resume_obj = torch.load(resume_path, map_location=device)
        if isinstance(resume_obj, dict) and "model" in resume_obj:
            load_state_dict_compat(model=raw_model, state_dict=resume_obj["model"])
            if "optimizer" in resume_obj and isinstance(resume_obj["optimizer"], dict):
                optimizer.load_state_dict(resume_obj["optimizer"])
            if scheduler is not None and "scheduler" in resume_obj and resume_obj["scheduler"] is not None:
                scheduler.load_state_dict(resume_obj["scheduler"])
            if "best_val" in resume_obj:
                best_val = float(resume_obj["best_val"])
            if "best_epoch" in resume_obj:
                best_epoch = int(resume_obj["best_epoch"])
            if "no_improve" in resume_obj:
                no_improve = int(resume_obj["no_improve"])
            loaded_best = resume_obj.get("best_val_metrics", {})
            if isinstance(loaded_best, dict):
                best_val_metrics = loaded_best
            last_epoch = int(resume_obj.get("epoch", 0))
            start_epoch = max(1, last_epoch + 1)
            log(
                f"[resume] loaded train state from {resume_path}: "
                f"last_epoch={last_epoch} next_epoch={start_epoch} "
                f"best_{best_key}={best_val:.6f} best_epoch={best_epoch}"
            )
        elif isinstance(resume_obj, dict):
            load_state_dict_compat(model=raw_model, state_dict=resume_obj)
            log(f"[resume] loaded model-only state_dict from {resume_path}")
        else:
            raise ValueError(f"Unsupported resume checkpoint format: {type(resume_obj)}")

    if start_epoch > int(args.num_epochs):
        log(
            f"[resume] start_epoch={start_epoch} exceeds num_epochs={args.num_epochs}, "
            "skip training loop and run final eval only"
        )

    for epoch in range(start_epoch, args.num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            fixed_mask_prob=args.mask_prob,
            diffusion_steps=args.diffusion_steps,
            mask_ratio_min=args.mask_ratio_min,
            mask_ratio_max=args.mask_ratio_max,
            full_mask_ratio=args.full_mask_ratio,
            device=device,
            grad_clip=args.grad_clip,
            loss_w_seq_causal=args.loss_w_seq_causal,
            loss_w_clean_align=args.loss_w_clean_align,
            loss_w_bidir_aux=args.loss_w_bidir_aux,
            loss_w_bidir_clean_align=args.loss_w_bidir_clean_align,
            seq_ratio_gamma=float(args.seq_ratio_gamma),
            seq_ratio_eps=float(args.seq_ratio_eps),
            seq_ratio_clip_min=float(args.seq_ratio_clip_min),
            seq_ratio_clip_max=float(args.seq_ratio_clip_max),
            seq_ratio_normalize=bool(args.seq_ratio_normalize),
            layer_codebook_sizes=layer_codebook_sizes,
            slot_codebook_sizes=slot_codebook_sizes,
            slot_offsets=slot_offsets,
            show_progress=(is_main and (not is_dist or rank == 0)),
            progress_desc=f"train@{epoch:03d}",
        )

        cur_lr = optimizer.param_groups[0]["lr"]
        log(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_stats['total']:.6f} "
            f"lr={cur_lr:.6g} time={time.time()-t0:.1f}s"
        )

        if epoch % max(1, args.eval_every) == 0:
            val_metrics = run_eval(sequences=val_sequences, progress_desc=f"val@{epoch}")
            log(f"Val: {format_metrics(val_metrics)}")

            cur = float(val_metrics[best_key])
            improved = cur > best_val
            if improved:
                best_val = cur
                best_epoch = epoch
                no_improve = 0
                best_val_metrics = val_metrics
                if is_main:
                    torch.save(raw_model.state_dict(), save_path)
                    log(f"Saved best model to {save_path} ({best_key}={best_val:.4f}, epoch={best_epoch})")

                if is_main:
                    meta_obj = {
                        "best_epoch": int(best_epoch),
                        "best_key": best_key,
                        "best_val": best_val_metrics,
                    }
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta_obj, f, ensure_ascii=False, indent=2)
            else:
                no_improve += 1
                log(f"No improvement on {best_key}. early_stop_counter={no_improve}")

        if is_main:
            train_state_obj = {
                "epoch": int(epoch),
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                "best_val": float(best_val),
                "best_epoch": int(best_epoch),
                "no_improve": int(no_improve),
                "best_val_metrics": best_val_metrics,
                "best_key": best_key,
                "args": vars(args),
            }
            torch.save(train_state_obj, train_state_path)

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            log(f"[stop] early stop at epoch={epoch}, best_epoch={best_epoch}, best_{best_key}={best_val:.6f}")
            break

    if is_dist:
        dist.barrier()
    if os.path.exists(save_path):
        state = torch.load(save_path, map_location=device)
        load_state_dict_compat(model=raw_model, state_dict=state)
    final_test = run_eval(sequences=test_sequences, progress_desc="final-test", max_users=0)
    log(f"Final Test(best_ckpt@epoch{best_epoch}): {format_metrics(final_test)}")
    if is_main and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_obj = json.load(f)
        meta_obj["final_test"] = final_test
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_obj, f, ensure_ascii=False, indent=2)
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
