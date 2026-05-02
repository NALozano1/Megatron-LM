# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional pre-dispatch MoE token reordering (packing / scheduling).

This hook runs in `MoEAllGatherTokenDispatcher.dispatch_preprocess` **after** flattening
hidden states to `[num_tokens, hidden]`, with `routing_map` and `probs` aligned on the
same token dimension.

When `TransformerConfig.moe_pre_dispatch_packing_enabled` is False (default), behaviour
is unchanged from stock Megatron-Core.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch


def pre_dispatch_pack_tokens(
    config: Any,
    hidden_states: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply optional packing permutation before TP/EP dispatch.

    Args:
        config: `TransformerConfig` (or any object with optional
            `moe_pre_dispatch_packing_enabled: bool`).
        hidden_states: `[num_tokens, hidden]`
        routing_map: token–expert routing map, first dim `num_tokens`.
        probs: routing probabilities, first dim `num_tokens`.

    Returns:
        `(hidden_states, routing_map, probs)` — permuted together when packing is enabled.
    """
    if not getattr(config, "moe_pre_dispatch_packing_enabled", False):
        return hidden_states, routing_map, probs
    # Packing policy (e.g. sim `local_window`) will be applied here in a follow-up change.
    return hidden_states, routing_map, probs
