"""Globally capped Grid-SID beam search with GPU catalog constraints."""

from __future__ import annotations

from typing import Sequence

import torch

from gpu_catalog_constraints import allowed_token_mask, initial_catalog_state, select_and_assign
from model import CleanKVCache


def _select_cache(cache: CleanKVCache, indices: torch.Tensor) -> CleanKVCache:
    return CleanKVCache(
        k_per_layer=[value.index_select(0, indices) for value in cache.k_per_layer],
        v_per_layer=[value.index_select(0, indices) for value in cache.v_per_layer],
        clean_item_idx=cache.clean_item_idx,
        clean_layer_idx=cache.clean_layer_idx,
    )


def _position_confidence(log_probs: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
    confidence = log_probs.max(dim=-1).values
    confidence.masked_fill_(layer_ids.ge(0), torch.finfo(confidence.dtype).min)
    return confidence


@torch.inference_mode()
def global_capped_tuple_decode_next_item_batch_with_cache(
    model,
    base_cache_batch: CleanKVCache,
    num_layers: int,
    num_splits: int,
    item_idx: int,
    slot_codebook_sizes: Sequence[int],
    slot_offsets: Sequence[int],
    beam_size: int,
    decode_fill_plan: Sequence[int],
    topk: int,
    catalog_tokens: torch.Tensor,
    temperature: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    if beam_size <= 0 or topk <= 0:
        raise ValueError("beam_size and topk must be positive")
    fill_plan = [int(value) for value in decode_fill_plan]
    if any(value <= 0 for value in fill_plan) or sum(fill_plan) != int(num_splits):
        raise ValueError("decode_fill_plan must contain positive values summing to num_splits")
    total_slots = int(num_layers) * int(num_splits)
    if len(slot_codebook_sizes) != total_slots or len(slot_offsets) != total_slots:
        raise ValueError("slot layout does not match Grid-SID shape")
    if catalog_tokens.dim() != 2 or int(catalog_tokens.size(1)) != total_slots:
        raise ValueError(f"catalog_tokens must be [num_items,{total_slots}], got {tuple(catalog_tokens.shape)}")

    dev = torch.device(device)
    num_users = int(base_cache_batch.k_per_layer[0].size(0))
    if num_users == 0:
        return torch.empty((0, 0, num_layers, num_splits), dtype=torch.long, device=dev)

    scores = torch.zeros((num_users, 1), dtype=torch.float32, device=dev)
    full_sid = torch.full(
        (num_users, 1, int(num_layers), int(num_splits)), -1, dtype=torch.long, device=dev
    )
    catalog_state = initial_catalog_state(
        num_users=num_users, num_items=int(catalog_tokens.size(0)), device=dev
    )
    global_cache = base_cache_batch

    for layer_idx in range(int(num_layers)):
        current_beam = int(scores.size(1))
        layer_ids = torch.full(
            (num_users, current_beam, int(num_splits)), -1, dtype=torch.long, device=dev
        )
        final_cache_parent = None
        final_cache_beam = 0

        for plan_index, n_fill in enumerate(fill_plan):
            current_beam = int(scores.size(1))
            cache_beam = current_beam
            pending_cache_parent = torch.arange(cache_beam, dtype=torch.long, device=dev).unsqueeze(0).expand(
                num_users, -1
            )
            flat_layer_ids = layer_ids.reshape(num_users * current_beam, int(num_splits))
            logits, local_offsets, _ = model.predict_generation_layer_local_logits(
                cache=global_cache,
                item_idx=int(item_idx),
                layer_idx=int(layer_idx),
                slot_codebook_sizes=slot_codebook_sizes,
                slot_offsets=slot_offsets,
                current_layer_ids=flat_layer_ids,
            )
            vocab_size = int(logits.size(-1))
            log_probs = torch.log_softmax(logits / float(temperature), dim=-1).view(
                num_users, current_beam, int(num_splits), vocab_size
            )
            top_positions = torch.topk(
                _position_confidence(log_probs=log_probs, layer_ids=layer_ids), k=int(n_fill), dim=2
            ).indices
            step_log_probs = log_probs

            for fill_index in range(int(n_fill)):
                current_beam = int(scores.size(1))
                positions = top_positions[:, :, fill_index]
                user_index = torch.arange(num_users, device=dev).unsqueeze(1)
                beam_index = torch.arange(current_beam, device=dev).unsqueeze(0)
                row_log_probs = step_log_probs[user_index, beam_index, positions, :]
                allowed = allowed_token_mask(
                    state=catalog_state,
                    positions=positions + int(layer_idx) * int(num_splits),
                    catalog_tokens=catalog_tokens,
                    vocab_size=vocab_size,
                )
                row_log_probs.masked_fill_(~allowed, torch.finfo(row_log_probs.dtype).min)

                token_candidates = min(int(beam_size), vocab_size)
                token_scores, local_tokens = torch.topk(row_log_probs, k=token_candidates, dim=2, sorted=False)
                candidate_scores = (scores.unsqueeze(-1) + token_scores.to(torch.float32)).reshape(
                    num_users, current_beam * token_candidates
                )
                next_beam = min(int(beam_size), int(candidate_scores.size(1)))
                next_scores, candidate_index = torch.topk(candidate_scores, k=next_beam, dim=1, sorted=False)
                parent = torch.div(candidate_index, token_candidates, rounding_mode="floor")
                token_choice = torch.remainder(candidate_index, token_candidates)

                selected_positions = positions.gather(1, parent)
                selected_local_tokens = local_tokens.gather(
                    1, parent.unsqueeze(-1).expand(-1, -1, token_candidates)
                ).gather(2, token_choice.unsqueeze(-1)).squeeze(-1)
                selected_global_tokens = selected_local_tokens + local_offsets[selected_positions]
                layer_ids = layer_ids.gather(1, parent.unsqueeze(-1).expand(-1, -1, int(num_splits)))
                full_sid = full_sid.gather(
                    1,
                    parent.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, int(num_layers), int(num_splits)),
                )
                out_user = torch.arange(num_users, device=dev).unsqueeze(1)
                out_beam = torch.arange(next_beam, device=dev).unsqueeze(0)
                layer_ids[out_user, out_beam, selected_positions] = selected_global_tokens
                catalog_state = select_and_assign(
                    state=catalog_state,
                    parent=parent,
                    positions=selected_positions + int(layer_idx) * int(num_splits),
                    tokens=selected_local_tokens,
                    catalog_tokens=catalog_tokens,
                )
                pending_cache_parent = pending_cache_parent.gather(1, parent)
                step_log_probs = step_log_probs.gather(
                    1,
                    parent.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, int(num_splits), vocab_size),
                )
                top_positions = top_positions.gather(1, parent.unsqueeze(-1).expand(-1, -1, int(n_fill)))
                scores = next_scores

            if plan_index + 1 < len(fill_plan):
                flat_cache_parent = (
                    torch.arange(num_users, device=dev).unsqueeze(1) * cache_beam + pending_cache_parent
                ).reshape(-1)
                global_cache = _select_cache(global_cache, flat_cache_parent)
            else:
                final_cache_parent = pending_cache_parent
                final_cache_beam = cache_beam

        full_sid[:, :, layer_idx, :] = layer_ids
        if layer_idx + 1 < int(num_layers):
            flat_final_parent = (
                torch.arange(num_users, device=dev).unsqueeze(1) * int(final_cache_beam) + final_cache_parent
            ).reshape(-1)
            global_cache = model.append_generation_layer_from_parent(
                cache=global_cache,
                parent_idx=flat_final_parent,
                layer_ids=layer_ids.reshape(num_users * int(scores.size(1)), int(num_splits)),
                item_idx=int(item_idx),
                layer_idx=int(layer_idx),
            )

    result_count = min(int(topk), int(scores.size(1)))
    _, order = torch.topk(scores, k=result_count, dim=1, sorted=True)
    return full_sid.gather(
        1, order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, int(num_layers), int(num_splits))
    )
