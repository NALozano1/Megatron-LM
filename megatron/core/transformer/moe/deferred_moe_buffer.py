# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Defer MoE preprocess/dispatch to accumulate routed tokens (global pool or binned coarse keys).

This module is CPU-light bookkeeping around GPU tensors — no Megatron dispatcher calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, Tuple, Union

import torch

MoEDeferredBatchMode = Literal['global_pool', 'binned']
MoEBinKeyMode = Literal['exact_topk', 'primary_expert', 'ep_rank_set_hash']


@dataclass(frozen=True)
class MoEDrainedPayload:
    """One drained batch feeding `MoELayer.preprocess(hidden, probs, routing_map)`."""

    hidden_states: torch.Tensor
    probs: torch.Tensor
    routing_map: torch.Tensor
    seq_span_start: int
    seq_span_end_exclusive: int
    batch_size: int


def _token_count_piece(seq_len: int, batch_size: int) -> int:
    return seq_len * batch_size


def _routing_row_key(row_mask: torch.Tensor, mode: MoEBinKeyMode, num_experts: int, ep_size: int) -> Union[int, Tuple[int, ...]]:
    """Discrete key derived from first logical expert mask row (already on-device tensor)."""
    if mode == 'exact_topk':
        idx = torch.nonzero(row_mask, as_tuple=False).view(-1)
        # tuple avoids numpy dependence
        return tuple(int(x) for x in idx.detach().tolist())
    if mode == 'primary_expert':
        # argmax on bool promotes; use probabilities from caller not available — use smallest True index.
        nz = torch.nonzero(row_mask, as_tuple=False)
        if nz.numel() == 0:
            return (-1,)
        return (int(nz[0].item()),)
    if mode == 'ep_rank_set_hash':
        assert ep_size >= 1
        nz = torch.nonzero(row_mask, as_tuple=False).view(-1)
        if nz.numel() == 0:
            return (-1,)
        experts_per_rank = max(num_experts // ep_size, 1)
        ranks = sorted({int(e.item()) // experts_per_rank for e in nz})
        return tuple(ranks)
    raise ValueError(f'Unknown MoE bin key mode {mode}')


def _piece_bin_key_from_routing(
    routing_map: torch.Tensor,
    mode: MoEBinKeyMode,
    num_experts: int,
    ep_size: int,
) -> Union[int, Tuple[int, ...]]:
    """Route tensor [T,E] coarse key: FIRST token row — entire strip routed into same coarse bin."""
    return _routing_row_key(routing_map[0], mode, num_experts, ep_size)


@dataclass
class _PendingPiece:
    seq_start: int
    seq_end_exclusive: int
    hidden_states: torch.Tensor  # [s,B,H]
    probs: torch.Tensor
    routing_map: torch.Tensor


def _sorted_contiguous_piece_groups(pieces: Sequence[_PendingPiece]) -> List[List[_PendingPiece]]:
    """Group pending span pieces into maximal contiguous `[seq_lo, seq_hi)` runs."""
    if not pieces:
        return []
    sort_p = sorted(pieces, key=lambda x: x.seq_start)
    groups: List[List[_PendingPiece]] = []
    cur: List[_PendingPiece] = [sort_p[0]]
    for p in sort_p[1:]:
        prev = cur[-1]
        if p.seq_start != prev.seq_end_exclusive:
            groups.append(cur)
            cur = [p]
        else:
            cur.append(p)
    groups.append(cur)
    return groups


def _merge_sorted_pieces_to_payload(pieces: Sequence[_PendingPiece]) -> MoEDrainedPayload:
    if not pieces:
        raise ValueError('merge requires non-empty sequence')
    sort_p = sorted(pieces, key=lambda x: x.seq_start)
    if len({p.hidden_states.size(1) for p in sort_p}) != 1:
        raise ValueError('inconsistent batch size across pending MoE deferred pieces')
    batch_size = sort_p[0].hidden_states.size(1)

    spans = [(p.seq_start, p.seq_end_exclusive) for p in sort_p]
    for (_, e0), (s1, _) in zip(spans[:-1], spans[1:]):
        if e0 != s1:
            raise ValueError(
                'non-contiguous deferred MoE spans after sort (need partition cover same forward)'
            )

    hid = torch.cat([p.hidden_states for p in sort_p], dim=0)
    probs = torch.cat([p.probs for p in sort_p], dim=0)
    routing_map = torch.cat([p.routing_map for p in sort_p], dim=0)

    return MoEDrainedPayload(
        hidden_states=hid,
        probs=probs,
        routing_map=routing_map,
        seq_span_start=sort_p[0].seq_start,
        seq_span_end_exclusive=sort_p[-1].seq_end_exclusive,
        batch_size=batch_size,
    )


class DeferredMoEBuffer:
    """Router-side buffer; produces drained payloads for expert dispatch."""

    def __init__(
        self,
        batch_mode: MoEDeferredBatchMode,
        target_tokens_global: int,
        bin_target_tokens: int,
        bin_key_mode: MoEBinKeyMode,
        num_experts: int,
        ep_parallel_size: int,
    ) -> None:
        self.batch_mode = batch_mode
        self.target_tokens_global = int(target_tokens_global)
        self.bin_target_tokens = int(bin_target_tokens)
        self.bin_key_mode = bin_key_mode
        self.num_experts = int(num_experts)
        self.ep_parallel_size = max(int(ep_parallel_size), 1)

        self._global_pending: List[_PendingPiece] = []
        self._global_token_rows: int = 0

        self._bins: Dict[Any, List[_PendingPiece]] = {}
        self._bin_tokens: Dict[Any, int] = {}

    def clear(self) -> None:
        self._global_pending.clear()
        self._global_token_rows = 0
        self._bins.clear()
        self._bin_tokens.clear()

    # --- internals ---------------------------------------------------------

    def _push_piece_to_global(self, piece: _PendingPiece) -> None:
        tc = _token_count_piece(piece.seq_end_exclusive - piece.seq_start, piece.hidden_states.size(1))
        self._global_pending.append(piece)
        self._global_token_rows += tc

    def _push_piece_to_binned(self, piece: _PendingPiece) -> None:
        key = _piece_bin_key_from_routing(
            piece.routing_map, self.bin_key_mode, self.num_experts, self.ep_parallel_size
        )
        self._bins.setdefault(key, []).append(piece)
        self._bin_tokens[key] = self._bin_tokens.get(key, 0) + _token_count_piece(
            piece.seq_end_exclusive - piece.seq_start, piece.hidden_states.size(1)
        )

    def push_routed_piece(
        self,
        seq_start: int,
        seq_end_exclusive: int,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> None:
        if seq_end_exclusive <= seq_start:
            return
        piece = _PendingPiece(
            seq_start=seq_start,
            seq_end_exclusive=seq_end_exclusive,
            hidden_states=hidden_states,
            probs=probs,
            routing_map=routing_map,
        )
        if self.batch_mode == 'global_pool':
            self._push_piece_to_global(piece)
        else:
            self._push_piece_to_binned(piece)

    def total_pending_token_rows_global(self) -> int:
        if self.batch_mode == 'global_pool':
            return self._global_token_rows
        return sum(self._bin_tokens.values())

    def drain_ready_payloads_mid_forward(self) -> List[MoEDrainedPayload]:
        """Emit preemptive payloads when thresholds are crossed (within one MoE forward)."""
        payloads: List[MoEDrainedPayload] = []
        if self.batch_mode == 'global_pool':
            if self._global_token_rows >= max(self.target_tokens_global, 1):
                for chunk in _sorted_contiguous_piece_groups(self._global_pending):
                    payloads.append(_merge_sorted_pieces_to_payload(chunk))
                self._global_pending.clear()
                self._global_token_rows = 0
            return payloads

        # binned mode: flush every bin eligible
        flushed_keys = [k for k, n in self._bin_tokens.items() if n >= max(self.bin_target_tokens, 1)]
        for k in flushed_keys:
            pieces = self._bins.pop(k)
            self._bin_tokens.pop(k, None)
            for chunk in _sorted_contiguous_piece_groups(pieces):
                payloads.append(_merge_sorted_pieces_to_payload(chunk))
        return payloads

    def drain_all_remaining(self) -> List[MoEDrainedPayload]:
        """Flush leftovers (normally at end-of-forward when `moe_deferred_flush_tokens_at_forward_end`)."""
        payloads: List[MoEDrainedPayload] = []
        if self.batch_mode == 'global_pool':
            if self._global_pending:
                for chunk in _sorted_contiguous_piece_groups(self._global_pending):
                    payloads.append(_merge_sorted_pieces_to_payload(chunk))
                self._global_pending.clear()
                self._global_token_rows = 0
            return payloads

        for k in list(self._bins.keys()):
            pieces = self._bins.pop(k)
            self._bin_tokens.pop(k, None)
            for chunk in _sorted_contiguous_piece_groups(pieces):
                payloads.append(_merge_sorted_pieces_to_payload(chunk))
        return payloads
