#!/usr/bin/env python3
"""Dataset utilities for SBD."""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


SplitDict = Dict[str, List]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_default_paths(
    dataset: str,
    sid_dir_name: str = "grid_sid",
) -> Dict[str, str]:
    root = f"data/{dataset}"
    sid_dir = f"outputs/{dataset}/{sid_dir_name}"
    return {
        "train_split": f"{root}/split/split_train.pkl",
        "val_split": f"{root}/split/split_val.pkl",
        "test_split": f"{root}/split/split_test.pkl",
        "sid_json": f"{sid_dir}/asin2sid_uni.json",
        "sid_meta": f"{sid_dir}/meta.json",
    }


def _to_sid_matrix(entry) -> np.ndarray:
    arr = np.asarray(entry, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"SID entry must be [B,K] or [B], got {arr.shape}")
    return arr


def infer_sid_shape_and_vocab(asin2sid_json: str) -> Tuple[int, int, int]:
    asin2sid = load_json(asin2sid_json)
    max_global = -1
    b = k = -1
    for _, sid in asin2sid.items():
        arr = _to_sid_matrix(sid)
        if b < 0:
            b, k = int(arr.shape[0]), int(arr.shape[1])
        elif arr.shape != (b, k):
            raise ValueError(f"Inconsistent SID shape {arr.shape}, expected {(b, k)}")
        max_global = max(max_global, int(arr.max()))
    if b < 0 or k < 0:
        raise ValueError("asin2sid is empty")
    v_global = max_global + 1
    return b, k, v_global


def _parse_layer_codebook_sizes(
    meta: Dict[str, Any],
    num_layers: int,
    max_local_vocab: Optional[int],
) -> Optional[List[int]]:
    vals = None
    rpq_cfg = meta.get("rpq_cfg")
    if isinstance(rpq_cfg, dict) and isinstance(rpq_cfg.get("layer_codebook_sizes"), list):
        vals = rpq_cfg.get("layer_codebook_sizes")
    elif isinstance(meta.get("layer_codebook_sizes"), list):
        vals = meta.get("layer_codebook_sizes")

    if vals is None:
        return None
    if len(vals) != int(num_layers):
        return None

    try:
        out = [int(v) for v in vals]
    except Exception:
        return None
    if any(v <= 0 for v in out):
        return None
    if max_local_vocab is not None and any(v > int(max_local_vocab) for v in out):
        return None
    return out


def load_sid_meta_or_infer(sid_meta_path: str, asin2sid_json: str) -> Dict[str, Any]:
    if not os.path.exists(sid_meta_path):
        raise FileNotFoundError(f"Missing sid_meta: {sid_meta_path}")

    meta = load_json(sid_meta_path)
    required = ["B", "K", "V_global"]
    if not all(k in meta for k in required):
        raise ValueError(f"sid_meta missing required keys {required}: {sid_meta_path}")

    b = int(meta["B"])
    k = int(meta["K"])
    v_global = int(meta["V_global"])

    layer_caps = _parse_layer_codebook_sizes(
        meta=meta,
        num_layers=b,
        max_local_vocab=None,
    )
    if layer_caps is None:
        raise ValueError(
            "sid_meta must include valid layer_codebook_sizes (top-level or rpq_cfg.layer_codebook_sizes)"
        )

    slot_sizes_raw = meta.get("slot_codebook_sizes")
    slot_offsets_raw = meta.get("slot_offsets")
    if not (isinstance(slot_sizes_raw, list) and len(slot_sizes_raw) == b * k):
        raise ValueError("sid_meta must include slot_codebook_sizes with length B*K")
    if not (isinstance(slot_offsets_raw, list) and len(slot_offsets_raw) == b * k):
        raise ValueError("sid_meta must include slot_offsets with length B*K")

    slot_sizes = [int(v) for v in slot_sizes_raw]
    slot_offsets = [int(v) for v in slot_offsets_raw]
    if any(v <= 0 for v in slot_sizes):
        raise ValueError(f"Invalid slot_codebook_sizes={slot_sizes}")
    if any(v < 0 for v in slot_offsets):
        raise ValueError(f"Invalid slot_offsets={slot_offsets}")

    return {
        "B": b,
        "K": k,
        "V_global": v_global,
        "layer_codebook_sizes": layer_caps,
        "slot_codebook_sizes": slot_sizes,
        "slot_offsets": slot_offsets,
    }


class SBDDataset(Dataset):

    def __init__(
        self,
        split_path: str,
        asin2sid_json: str,
        strict: bool = True,
        max_seq_len: int = 0,
    ):
        super().__init__()
        split: SplitDict = load_pkl(split_path)
        self.user_ids: List[str] = split["user_ids"]
        self.max_seq_len = int(max_seq_len)
        if self.max_seq_len > 0:
            self.user_seqs = [seq[-self.max_seq_len :] for seq in split["user_seqs"]]
        else:
            self.user_seqs = split["user_seqs"]
        raw_asin2sid = load_json(asin2sid_json)
        if len(raw_asin2sid) == 0:
            raise ValueError("asin2sid is empty")
        self.asin2sid: Dict[str, np.ndarray] = {}
        self.num_layers = -1
        self.num_splits = -1
        for asin, sid in raw_asin2sid.items():
            sid_arr = _to_sid_matrix(sid).astype(np.int64, copy=False)
            if self.num_layers < 0:
                self.num_layers = int(sid_arr.shape[0])
                self.num_splits = int(sid_arr.shape[1])
            elif sid_arr.shape != (self.num_layers, self.num_splits):
                raise ValueError(f"Inconsistent SID shape {sid_arr.shape}, expected {(self.num_layers, self.num_splits)}")
            self.asin2sid[asin] = sid_arr
        self.strict = bool(strict)
        self._zero_sid = np.zeros((self.num_layers, self.num_splits), dtype=np.int64)

        if self.strict:
            self._validate_coverage()

    def _validate_coverage(self):
        missing = []
        for uid, seq in zip(self.user_ids, self.user_seqs):
            for asin in seq:
                if asin not in self.asin2sid:
                    missing.append((uid, asin))
                    if len(missing) >= 10:
                        break
            if len(missing) >= 10:
                break
        if missing:
            preview = ", ".join([f"({u},{a})" for u, a in missing[:5]])
            raise KeyError(
                "Found ASINs in split not present in asin2sid. "
                f"First cases: {preview}. You can set strict=False to bypass."
            )

    def __len__(self) -> int:
        return len(self.user_seqs)

    def _lookup_sid(self, asin: str) -> np.ndarray:
        if asin not in self.asin2sid:
            if self.strict:
                raise KeyError(f"ASIN missing in asin2sid: {asin}")
            return self._zero_sid
        return self.asin2sid[asin]

    def __getitem__(self, index: int):
        seq = self.user_seqs[index]
        sid_seq = np.stack([self._lookup_sid(a) for a in seq], axis=0).astype(np.int64)  # [n,B,K]
        n = sid_seq.shape[0]
        item_mask = np.ones((n,), dtype=bool)
        return {
            "user_id": self.user_ids[index],
            "sid_ids": torch.from_numpy(sid_seq),
            "item_mask": torch.from_numpy(item_mask),
            "seq_len": int(n),
        }


class SBDCollator:

    def __call__(self, batch: List[Dict]):
        bsz = len(batch)
        max_n = max(int(x["seq_len"]) for x in batch)
        b_rq = int(batch[0]["sid_ids"].shape[1])
        k_pq = int(batch[0]["sid_ids"].shape[2])

        sid_ids = torch.zeros((bsz, max_n, b_rq, k_pq), dtype=torch.long)
        item_mask = torch.zeros((bsz, max_n), dtype=torch.bool)
        seq_len = torch.zeros((bsz,), dtype=torch.long)

        user_ids = []
        for i, x in enumerate(batch):
            n = int(x["seq_len"])
            sid_ids[i, :n] = x["sid_ids"]
            item_mask[i, :n] = True
            seq_len[i] = n
            user_ids.append(x["user_id"])

        return {
            "user_id": user_ids,
            "sid_ids": sid_ids,
            "item_mask": item_mask,
            "seq_len": seq_len,
        }
