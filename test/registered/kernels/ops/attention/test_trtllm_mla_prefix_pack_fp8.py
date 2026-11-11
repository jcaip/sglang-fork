"""trtllm_mla pack_prefix_chunk_kv is bit-identical to the path it replaces:
forward_mha's BF16 K assembly followed by _quantize_fp8_qkv's unit-scale cast."""

import sys

import pytest
import torch

from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    _quantize_fp8_qkv,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

DEVICE = "cuda"
FP8 = torch.float8_e4m3fn
DN, DR, DV = 128, 64, 128  # qk_nope, qk_rope, v head dims of DeepSeek-style MLA


def _backend():
    backend = object.__new__(TRTLLMMLABackend)
    backend.data_type = FP8
    backend.q_data_type = torch.bfloat16
    backend.forward_prefill_metadata = None
    backend._init_pack_prefix_chunk_kv()
    assert backend.pack_prefix_chunk_kv is not None
    return backend


def _prefix_kv(num_tokens, num_heads, k_pe_dim=3, scale=1.0):
    """k_nope and v as strided views of a kv_b_proj output, as forward_mha does."""
    torch.manual_seed(num_tokens)
    kv = torch.randn(num_tokens, num_heads, DN + DV, device=DEVICE) * scale
    k_pe = torch.randn(num_tokens, 1, DR, device=DEVICE) * scale
    kv, k_pe = kv.to(torch.bfloat16), k_pe.to(torch.bfloat16)
    if k_pe_dim == 2:
        k_pe = k_pe.squeeze(1)
    return kv[..., :DN], k_pe, kv[..., DN:]


def _reference(k_nope, k_pe, v):
    """forward_mha without a pack hook, then _quantize_fp8_qkv at unit scale."""
    num_heads = k_nope.shape[1]
    k = torch.empty(
        (k_nope.shape[0], num_heads, DN + DR), dtype=v.dtype, device=v.device
    )
    k[..., :DN] = k_nope
    k[..., DN:] = k_pe if k_pe.dim() == 3 else k_pe.unsqueeze(1)
    q = torch.zeros(1, num_heads, DN + DR, dtype=v.dtype, device=v.device)
    layer = type("Layer", (), {"k_scale_float": None, "v_scale_float": None})()
    _, k, v, _, _ = _quantize_fp8_qkv(q, k, v.view(-1, num_heads, DV), layer)
    return k, v


def _assert_bit_equal(actual, expected):
    assert actual.dtype == expected.dtype == FP8
    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("num_heads", [12, 128])
# Covers every mla_kv_pack_quantize_fp8 dispatch bucket (s<=2, <=16, <=192, <=1536, larger).
@pytest.mark.parametrize("num_tokens", [1, 15, 100, 257, 4096])
@pytest.mark.parametrize("k_pe_dim", [2, 3])
def test_bit_exact_vs_bf16_pack_and_cast(num_heads, num_tokens, k_pe_dim):
    k_nope, k_pe, v = _prefix_kv(num_tokens, num_heads, k_pe_dim)
    k, v8 = _backend().pack_prefix_chunk_kv(k_nope, k_pe, v)
    ref_k, ref_v = _reference(k_nope, k_pe, v)
    _assert_bit_equal(k, ref_k)
    _assert_bit_equal(v8, ref_v)


def test_bit_exact_long_prefix():
    """One ~100K-token prefix chunk at 12 local heads."""
    k_nope, k_pe, v = _prefix_kv(99840, 12)
    k, v8 = _backend().pack_prefix_chunk_kv(k_nope, k_pe, v)
    ref_k, ref_v = _reference(k_nope, k_pe, v)
    _assert_bit_equal(k, ref_k)
    _assert_bit_equal(v8, ref_v)


def test_bit_exact_saturation_subnormals_inf_nan():
    """Out-of-range, tie, subnormal, inf and NaN inputs round like torch's
    CUDA cast (RNE, saturating to +-448)."""
    k_nope, k_pe, v = _prefix_kv(4096, 12, scale=200.0)
    special = torch.tensor(
        [448, 456, 464, 465, 472, 480, 512, 1e4, 3e38]
        + [2.0**-9, 2.0**-10, 2.0**-7 * 1.5, 1e-30, 0.0]
        + [float("inf"), float("nan")],
        device=DEVICE,
    ).to(torch.bfloat16)
    special = torch.cat([special, -special])
    k_nope[:64, 0, : special.numel()] = special
    v[:64, 3, : special.numel()] = special
    k_pe[:8, 0, : special.numel()] = special

    k, v8 = _backend().pack_prefix_chunk_kv(k_nope, k_pe, v)
    ref_k, ref_v = _reference(k_nope, k_pe, v)
    _assert_bit_equal(k, ref_k)
    _assert_bit_equal(v8, ref_v)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
