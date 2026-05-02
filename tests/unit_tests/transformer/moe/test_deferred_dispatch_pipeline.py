# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Phase tests: deferred dispatch config, buffer logic, MoELayer parity (CUDA)."""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.deferred_moe_buffer import DeferredMoEBuffer
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestPhase2DeferredConfigFields:
    def test_defaults_off(self):
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            num_moe_experts=4,
            moe_ffn_hidden_size=16,
            add_bias_linear=False,
        )
        assert cfg.moe_deferred_dispatch_enabled is False
        assert cfg.moe_deferred_batch_mode == 'global_pool'
        assert cfg.moe_deferred_target_tokens == 4096
        assert cfg.moe_deferred_sequence_strip_count == 1


class TestPhase3DeferredMoEBuffer:
    def test_global_mid_drain_then_end(self):
        B, H = 2, 3
        buf = DeferredMoEBuffer(
            batch_mode='global_pool',
            target_tokens_global=8,
            bin_target_tokens=10,
            bin_key_mode='primary_expert',
            num_experts=4,
            ep_parallel_size=1,
        )
        rm_1tok = torch.zeros(B, 4, dtype=torch.bool)
        rm_1tok[:, 0] = True
        buf.push_routed_piece(0, 1, torch.randn(1, B, H), torch.randn(B, 4), rm_1tok)
        assert buf.total_pending_token_rows_global() == B
        assert buf.drain_ready_payloads_mid_forward() == []

        rm_2tok = torch.zeros(2 * B, 4, dtype=torch.bool)
        rm_2tok[:, 0] = True
        buf.push_routed_piece(1, 3, torch.randn(2, B, H), torch.randn(2 * B, 4), rm_2tok)
        assert buf.total_pending_token_rows_global() == 3 * B
        assert buf.drain_ready_payloads_mid_forward() == []

        buf.push_routed_piece(3, 4, torch.randn(1, B, H), torch.randn(B, 4), rm_1tok.clone())
        assert buf.total_pending_token_rows_global() == 4 * B
        mid = buf.drain_ready_payloads_mid_forward()
        assert len(mid) == 1
        assert mid[0].hidden_states.shape[0] == 4
        assert buf.total_pending_token_rows_global() == 0

        rm_3tok = torch.zeros(3 * B, 4, dtype=torch.bool)
        rm_3tok[:, 0] = True
        buf.push_routed_piece(4, 7, torch.randn(3, B, H), torch.randn(3 * B, 4), rm_3tok)
        tail = buf.drain_all_remaining()
        assert len(tail) == 1
        assert tail[0].seq_span_end_exclusive - tail[0].seq_span_start == 3

    def test_binned_splits_non_contiguous_bins(self):
        B, H = 2, 2
        buf = DeferredMoEBuffer(
            batch_mode='binned',
            target_tokens_global=999,
            bin_target_tokens=10,
            bin_key_mode='primary_expert',
            num_experts=8,
            ep_parallel_size=2,
        )
        rm_a = torch.zeros(2 * B, 8, dtype=torch.bool)
        rm_a[:, 0] = True
        rm_b = torch.zeros(2 * B, 8, dtype=torch.bool)
        rm_b[:, 1] = True
        # same coarse primary key triggers same bin ONLY if routing_map[0] matches — different columns => different bins
        buf.push_routed_piece(0, 2, torch.ones(2, B, H), torch.ones(2 * B, 8), rm_a)
        buf.push_routed_piece(4, 6, torch.ones(2, B, H) * 2, torch.ones(2 * B, 8) * 2, rm_b)
        out = buf.drain_all_remaining()
        assert len(out) >= 2  # non-contiguous spans create separate payloads


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA deferred MoE parity test runs on GPU')
class TestPhase4DeferredMoEForwardParity:
    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_deferred_strip1_matches_stock(self):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        num_experts = 4
        hidden_size = 12
        seq_length = 8
        micro_batch_size = 2

        common = dict(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type='allgather',
            moe_router_load_balancing_type='aux_loss',
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=128,
            add_bias_linear=False,
        )

        cfg_off = TransformerConfig(**common)
        cfg_def = TransformerConfig(
            **common,
            moe_deferred_dispatch_enabled=True,
            moe_deferred_batch_mode='global_pool',
            moe_deferred_target_tokens=1_000_000,
            moe_deferred_sequence_strip_count=1,
            moe_pre_dispatch_packing_enabled=False,
        )

        subs = get_gpt_layer_local_submodules(num_experts=num_experts, moe_grouped_gemm=False)
        layer_off = MoELayer(cfg_off, subs.mlp.submodules).cuda().eval()
        layer_def = MoELayer(cfg_def, subs.mlp.submodules).cuda().eval()

        sd = layer_off.state_dict()
        layer_def.load_state_dict(sd)

        inp = torch.randn(
            seq_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.float32,
        )

        torch.manual_seed(0)
        with torch.no_grad():
            out_off, _ = layer_off(inp)
        torch.manual_seed(0)
        with torch.no_grad():
            out_def, _ = layer_def(inp)

        assert torch.allclose(out_off, out_def, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA only')
class TestPhase4DeferredMultiStripSmoke:
    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_three_routing_strips_runs_and_finite(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(99)
        num_experts = 4
        hidden_size = 16

        common = dict(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type='allgather',
            moe_router_load_balancing_type='aux_loss',
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=64,
            add_bias_linear=False,
            fp16=True,
            params_dtype=torch.float16,
            moe_deferred_dispatch_enabled=True,
            moe_deferred_target_tokens=1_000_000,
            moe_deferred_sequence_strip_count=3,
        )
        cfg = TransformerConfig(**common)

        subs = get_gpt_layer_local_submodules(num_experts=num_experts, moe_grouped_gemm=False)
        layer = MoELayer(cfg, subs.mlp.submodules).cuda().eval()

        inp = torch.randn(9, 2, hidden_size, device='cuda', dtype=torch.float16)
        with torch.no_grad():
            out, _ = layer(inp)
        assert out.shape == inp.shape
        assert torch.isfinite(out).all().item()

