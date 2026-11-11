"""trtllm_mla pack_prefix_chunk_kv: where the FP8 prefix pack is exposed, the
batches that keep the model-dtype layout, and the unit-scale guard in
_quantize_fp8_qkv. Bit-exactness of the FP8 pack itself needs a GPU; see
test/registered/kernels/ops/attention/test_trtllm_mla_prefix_pack_fp8.py."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.attention.cutedsl_mla_backend import (  # noqa: E402
    CuteDslMLAPrefillBackend,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (  # noqa: E402
    TRTLLMMLABackend,
    TRTLLMMLAPrefillMetadata,
    _quantize_fp8_qkv,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

FP8 = torch.float8_e4m3fn
H, DN, DR, DV = 4, 16, 8, 16


def _backend(cls=TRTLLMMLABackend, *, kv_dtype=FP8, model_dtype=torch.bfloat16):
    backend = object.__new__(cls)
    backend.data_type = kv_dtype
    backend.q_data_type = model_dtype
    backend.forward_prefill_metadata = None
    backend._init_pack_prefix_chunk_kv()
    return backend


def _prefix_kv(num_tokens, k_pe_dim=3):
    """k_nope and v as strided views of a kv_b_proj output, as forward_mha does."""
    torch.manual_seed(num_tokens)
    kv = torch.randn(num_tokens, H, DN + DV).to(torch.bfloat16)
    k_pe = torch.randn(num_tokens, 1, DR).to(torch.bfloat16)
    if k_pe_dim == 2:
        k_pe = k_pe.squeeze(1)
    return kv[..., :DN], k_pe, kv[..., DN:]


class TestPackHookExposure(CustomTestCase):
    def test_exposed_only_for_fp8_kv_with_bf16_model(self):
        # forward_mha probes getattr(backend, "pack_prefix_chunk_kv", None); a
        # visible hook also flips the latent read dtype to BF16, so an FP16
        # model or a BF16 KV cache must not see it.
        cases = [
            (FP8, torch.bfloat16, False, True),
            (FP8, torch.bfloat16, True, False),
            (FP8, torch.float16, False, False),
            (torch.bfloat16, torch.bfloat16, False, False),
        ]
        for kv_dtype, model_dtype, disabled, exposed in cases:
            with (
                self.subTest(kv=kv_dtype, model=model_dtype, disabled=disabled),
                envs.SGLANG_DISABLE_TRTLLM_MLA_FP8_PREFIX_PACK.override(disabled),
            ):
                backend = _backend(kv_dtype=kv_dtype, model_dtype=model_dtype)
                hook = getattr(backend, "pack_prefix_chunk_kv", None)
                self.assertEqual(hook is not None, exposed)

    def test_subclass_override_is_not_hidden(self):
        # CuteDslMLAPrefillBackend ships its own FP8 pack; the base gate must
        # not shadow it with an instance attribute.
        with envs.SGLANG_DISABLE_TRTLLM_MLA_FP8_PREFIX_PACK.override(True):
            backend = _backend(CuteDslMLAPrefillBackend)
        self.assertIsNotNone(getattr(backend, "pack_prefix_chunk_kv", None))


class TestFlashInferFallbackLayout(CustomTestCase):
    def test_fallback_batch_keeps_model_dtype_layout(self):
        # Batches forward_extend hands to the FlashInfer MLA parent (captured
        # prefill graph, or chunked prefix cache disabled) must get BF16 K/V in
        # the layout forward_mha builds without a pack hook.
        backend = _backend()
        backend.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
            max_seq_len=1,
            cum_seq_lens=torch.zeros(2, dtype=torch.int32),
            seq_lens=torch.ones(1, dtype=torch.int32),
            fallback_to_flashinfer_impl=True,
        )
        for k_pe_dim in (2, 3):
            with self.subTest(k_pe_dim=k_pe_dim):
                k_nope, k_pe, v = _prefix_kv(7, k_pe_dim)
                k, v_out = backend.pack_prefix_chunk_kv(k_nope, k_pe, v)

                pe = k_pe if k_pe_dim == 3 else k_pe.unsqueeze(1)
                expected_k = torch.cat([k_nope, pe.expand(-1, H, -1)], dim=-1)
                self.assertEqual(k.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(k, expected_k))
                self.assertIs(v_out, v)


class TestQuantizeFp8QkvUnitScaleGuard(CustomTestCase):
    def _qkv(self):
        q = torch.randn(3, H, DN + DR).to(torch.bfloat16)
        k = torch.randn(5, H, DN + DR).to(FP8)
        v = torch.randn(5, H, DV).to(FP8)
        return q, k, v

    def test_unit_scale_passes_packed_kv_through(self):
        # The packed tensors reach the ragged kernel untouched, with unit
        # dequant scales, so the pack is the only K/V cast.
        q, k, v = self._qkv()
        for k_scale, v_scale in ((None, None), (1.0, 1.0)):
            layer = SimpleNamespace(k_scale_float=k_scale, v_scale_float=v_scale)
            q8, k8, v8, ks, vs = _quantize_fp8_qkv(q, k, v, layer)
            self.assertEqual(q8.dtype, FP8)
            self.assertIs(k8, k)
            self.assertIs(v8, v)
            self.assertEqual((ks, vs), (1.0, 1.0))

    def test_non_unit_scale_with_packed_kv_raises(self):
        # The scaled path would requantize values already rounded at unit
        # scale; fail loudly instead of attending with the wrong K/V.
        q, k, v = self._qkv()
        for k_scale, v_scale in ((0.5, 1.0), (1.0, 2.0)):
            layer = SimpleNamespace(k_scale_float=k_scale, v_scale_float=v_scale)
            with self.subTest(k_scale=k_scale, v_scale=v_scale):
                with self.assertRaisesRegex(
                    RuntimeError, "SGLANG_DISABLE_TRTLLM_MLA_FP8_PREFIX_PACK"
                ):
                    _quantize_fp8_qkv(q, k, v, layer)


if __name__ == "__main__":
    unittest.main()
