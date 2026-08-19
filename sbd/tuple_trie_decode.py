"""GPU-constrained Grid-SID decoding and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from global_beam_decode import global_capped_tuple_decode_next_item_batch_with_cache


@dataclass
class TrieNode:
    children: Dict[Tuple[int, ...], "TrieNode"]
    asins: List[str]

    def __init__(self):
        self.children = {}
        self.asins = []

    def add_child(self, edge: Tuple[int, ...]) -> "TrieNode":
        if edge not in self.children:
            self.children[edge] = TrieNode()
        return self.children[edge]


def _sid_to_matrix(entry) -> np.ndarray:
    arr = np.asarray(entry, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"SID must be [B,K] or [B], got {arr.shape}")
    return arr


def build_tuple_trie(asin2sid: Dict[str, Sequence]) -> Tuple[TrieNode, int, int]:
    root = TrieNode()
    b = k = None

    for asin, sid in asin2sid.items():
        sid_mat = _sid_to_matrix(sid)
        if b is None:
            b, k = sid_mat.shape
        elif sid_mat.shape != (b, k):
            raise ValueError(f"Inconsistent SID shape {sid_mat.shape}, expected {(b, k)}")

        node = root
        for layer in range(sid_mat.shape[0]):
            node = node.add_child(tuple(int(x) for x in sid_mat[layer].tolist()))
        node.asins.append(str(asin))

    if b is None or k is None:
        raise ValueError("asin2sid is empty")
    return root, int(b), int(k)


def build_catalog_tokens(
    asin2sid: Dict[str, Sequence],
    slot_offsets: Sequence[int],
    device: torch.device | str,
) -> torch.Tensor:
    rows = []
    for sid in asin2sid.values():
        flat_sid = [int(token) for layer in sid for token in layer]
        if len(flat_sid) != len(slot_offsets):
            raise ValueError(
                f"SID has {len(flat_sid)} tokens but slot_offsets has {len(slot_offsets)} entries"
            )
        rows.append([token - int(slot_offsets[pos]) for pos, token in enumerate(flat_sid)])
    return torch.tensor(rows, dtype=torch.long, device=device)


def _lookup_asins_for_sid_key(trie_root: TrieNode, sid_key: Sequence[Sequence[int]]) -> List[str]:
    node = trie_root
    for layer in sid_key:
        node = node.children.get(tuple(int(x) for x in layer))
        if node is None:
            return []
    return list(node.asins)


def _materialize_raw_sid_batch(
    trie_root: TrieNode,
    sid_batch: torch.Tensor,
) -> List[List[Dict]]:
    sid_cpu = sid_batch.detach().cpu()
    out: List[List[Dict]] = []
    for user_i in range(int(sid_cpu.size(0))):
        row: List[Dict] = []
        for cand_i in range(int(sid_cpu.size(1))):
            sid_row = sid_cpu[user_i, cand_i]
            if bool((sid_row < 0).any()):
                continue
            sid_key = tuple(
                tuple(int(x) for x in sid_row[layer_i].tolist())
                for layer_i in range(int(sid_row.size(0)))
            )
            asins = _lookup_asins_for_sid_key(trie_root=trie_root, sid_key=sid_key)
            if asins:
                row.append({"sid": [list(layer) for layer in sid_key], "asins": asins})
        out.append(row)
    return out


@torch.inference_mode()
def eval_recall_ndcg_next_item_trie(
    model,
    sequences: List[List[str]],
    asin2sid: Dict[str, Sequence],
    trie_root: TrieNode,
    num_layers: int,
    num_splits: int,
    slot_codebook_sizes: Optional[Sequence[int]],
    slot_offsets: Optional[Sequence[int]],
    topk_list: List[int],
    beam_size: int = 50,
    decode_fill_plan: Optional[Sequence[int]] = None,
    max_users: int = 0,
    temperature: float = 1.0,
    device: str = "cuda",
    cache_build_batch_size: int = 64,
    catalog_tokens: Optional[torch.Tensor] = None,
    show_progress: bool = True,
    progress_desc: str = "eval",
) -> Dict[str, float]:
    if cache_build_batch_size <= 0:
        raise ValueError("cache_build_batch_size must be > 0")
    if decode_fill_plan is None or len(decode_fill_plan) == 0:
        raise ValueError("decode_fill_plan must be provided for GPU decoding")
    if catalog_tokens is None:
        raise ValueError("catalog_tokens must be provided for GPU decoding")

    fill_plan = [int(value) for value in decode_fill_plan]
    if any(value <= 0 for value in fill_plan) or sum(fill_plan) != int(num_splits):
        raise ValueError(
            f"Strict fill plan requires positive entries summing to num_splits={num_splits}, got {fill_plan}"
        )
    topk_list = sorted(int(k) for k in topk_list)
    max_k = topk_list[-1]

    hit = {k: 0.0 for k in topk_list}
    ndcg = {k: 0.0 for k in topk_list}
    cnt = 0
    was_training = bool(model.training)
    model.eval()
    try:
        samples: List[Tuple[int, Sequence, str]] = []
        for seq in sequences:
            if max_users > 0 and len(samples) >= max_users:
                break
            if len(seq) < 2:
                continue
            history = seq[:-1]
            target = seq[-1]
            if target not in asin2sid or any(item not in asin2sid for item in history):
                continue
            hist_sid = [_sid_to_matrix(asin2sid[item]).tolist() for item in history]
            samples.append((len(history), hist_sid, str(target)))

        buckets: Dict[int, List[Tuple[Sequence, str]]] = {}
        for n_hist, hist_sid, target in samples:
            buckets.setdefault(n_hist, []).append((hist_sid, target))

        pbar = tqdm(total=len(samples), desc=progress_desc, leave=False, dynamic_ncols=True) if show_progress else None
        dev = torch.device(device)
        for n_hist, bucket in buckets.items():
            for start in range(0, len(bucket), cache_build_batch_size):
                chunk = bucket[start : start + cache_build_batch_size]
                hist_batch = torch.as_tensor([item[0] for item in chunk], dtype=torch.long, device=dev)
                cache_batch = model.build_generation_cache(history_sid_ids=hist_batch)
                raw_sid = global_capped_tuple_decode_next_item_batch_with_cache(
                    model=model,
                    base_cache_batch=cache_batch,
                    num_layers=num_layers,
                    num_splits=num_splits,
                    item_idx=n_hist,
                    slot_codebook_sizes=slot_codebook_sizes,
                    slot_offsets=slot_offsets,
                    beam_size=beam_size,
                    decode_fill_plan=fill_plan,
                    topk=max_k,
                    catalog_tokens=catalog_tokens,
                    temperature=temperature,
                    device=device,
                )
                predictions = _materialize_raw_sid_batch(trie_root=trie_root, sid_batch=raw_sid)

                for i, (_, target) in enumerate(chunk):
                    pred_items: List[str] = []
                    for candidate in predictions[i]:
                        pred_items.extend(str(item) for item in candidate.get("asins", []))
                        if len(pred_items) >= max_k:
                            break
                    found_rank = next((rank for rank, item in enumerate(pred_items) if item == target), None)
                    for k in topk_list:
                        if found_rank is not None and found_rank < k:
                            hit[k] += 1.0
                            ndcg[k] += 1.0 / np.log2(found_rank + 2)
                    cnt += 1
                if pbar is not None:
                    pbar.update(len(chunk))
        if pbar is not None:
            pbar.close()

        metrics = {}
        for k in topk_list:
            metrics[f"recall@{k}"] = float(hit[k] / max(1, cnt))
            metrics[f"ndcg@{k}"] = float(ndcg[k] / max(1, cnt))
        metrics["count"] = float(cnt)
        return metrics
    finally:
        if was_training:
            model.train()
