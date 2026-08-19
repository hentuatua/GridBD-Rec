from __future__ import annotations

import torch


def initial_catalog_state(num_users: int, num_items: int, device: torch.device) -> torch.Tensor:
    return torch.ones((int(num_users), 1, int(num_items)), dtype=torch.uint8, device=device)


def allowed_token_mask(
    state: torch.Tensor,
    positions: torch.Tensor,
    catalog_tokens: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    num_users, beam_size, num_items = state.shape
    token_ids = catalog_tokens.index_select(1, positions.reshape(-1)).transpose(0, 1).contiguous()
    allowed = torch.zeros(
        (int(num_users) * int(beam_size), int(vocab_size)), dtype=torch.uint8, device=state.device
    )
    allowed.scatter_reduce_(
        1, token_ids, state.reshape(-1, int(num_items)), reduce="amax", include_self=True
    )
    return allowed.view(int(num_users), int(beam_size), int(vocab_size)).bool()


def select_and_assign(
    state: torch.Tensor,
    parent: torch.Tensor,
    positions: torch.Tensor,
    tokens: torch.Tensor,
    catalog_tokens: torch.Tensor,
) -> torch.Tensor:
    num_users, next_beam = parent.shape
    num_items = int(state.size(2))
    selected = state.gather(1, parent.unsqueeze(-1).expand(-1, -1, num_items))
    catalog_at_position = catalog_tokens.index_select(1, positions.reshape(-1)).transpose(0, 1)
    catalog_at_position = catalog_at_position.reshape(int(num_users), int(next_beam), num_items)
    selected.mul_(catalog_at_position.eq(tokens.unsqueeze(-1)))
    return selected
