# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only tests for optional MoE `pre_dispatch_pack_tokens` hook."""

from types import SimpleNamespace

import torch

from megatron.core.transformer.moe.token_packing import pre_dispatch_pack_tokens


def test_pack_tokens_noop_when_disabled():
    cfg = SimpleNamespace(moe_pre_dispatch_packing_enabled=False)
    h = torch.randn(5, 8)
    r = torch.randint(0, 2, (5, 3), dtype=torch.bool)
    p = torch.randn(5, 3)
    h2, r2, p2 = pre_dispatch_pack_tokens(cfg, h, r, p)
    assert h2 is h and r2 is r and p2 is p


def test_pack_tokens_noop_when_flag_true_stub():
    cfg = SimpleNamespace(moe_pre_dispatch_packing_enabled=True)
    h = torch.randn(4, 16)
    r = torch.zeros(4, 2, dtype=torch.bool)
    p = torch.randn(4, 2)
    h2, r2, p2 = pre_dispatch_pack_tokens(cfg, h, r, p)
    assert torch.equal(h2, h) and torch.equal(r2, r) and torch.equal(p2, p)


def test_missing_config_attr_defaults_to_disabled():
    cfg = SimpleNamespace()
    h = torch.ones(2, 3)
    r = torch.zeros(2, 1, dtype=torch.bool)
    p = torch.randn(2, 1)
    h2, r2, p2 = pre_dispatch_pack_tokens(cfg, h, r, p)
    assert h2 is h
