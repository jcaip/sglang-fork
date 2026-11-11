"""SGLANG_CUTEDSL_MLA_DECODE_MIN_KV_SPLITS on the real FlashInfer CuTe DSL kernel.

At graph bs 40 FlashInfer's occupancy heuristic gives the Kimi K3 decode shape
(12 heads per GPU) one split per request. The floor must (1) pick 8 splits from
FlashInfer's own sizer wherever 8 fit the backend's 150 MB workspace, and
(2) return what the default ``trtllm_batch_decode_with_kv_cache_mla`` path
returns -- BF16 out of the same shape and a natural-log LSE -- up to split-KV
round-off. Split-KV reorders the softmax reduction, so this is not bit-exact;
the measured gap (0.7-1.3e-3 max abs on out) is the same scale as FlashInfer's
own split-2 path.

Runs on SM100 only, where the CuTe DSL MLA decode exists.

    python -m pytest test/registered/attention/test_cutedsl_mla_kv_split_floor_kernel.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.cutedsl_mla_backend import (
    CuteDslMLABackend,
    floor_cutedsl_mla_decode_kv_splits,
)
from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="4-gpu-b200")

_B = 40
_HEADS = 12
_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
_QK_NOPE_HEAD_DIM = 128
_PAGE_SIZE = 64
_DCP = 8
_FLOOR = 8
_WORKSPACE_BYTES = 150 * 1024 * 1024  # TRTLLMMLABackend DEFAULT_WORKSPACE_SIZE_MB
_SCALING = (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM) ** -0.5


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipUnless(is_sm100_supported(), "the CuTe DSL MLA decode requires SM100")
class TestCuteDslMLADecodeKVSplitFloor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flashinfer.cute_dsl.attention.monolithic import mla_decode

        cls.mla_decode = mla_decode
        cls.sizer = mla_decode._get_split_kv_and_workspace_size
        cls.num_sms = mla_decode.get_num_sm(torch.device("cuda"))
        cls.workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")

    def _backend(self, data_type):
        backend = CuteDslMLABackend.__new__(CuteDslMLABackend)
        backend._cutedsl_mla_decode = self.mla_decode
        backend.kv_lora_rank = _KV_LORA_RANK
        backend.qk_rope_head_dim = _QK_ROPE_HEAD_DIM
        backend.workspace_buffer = self.workspace
        backend.data_type = data_type
        return backend

    def test_floor_uses_flashinfer_sizer_and_fits_workspace(self):
        max_seq_len = 20_000
        for batch_size in (1, 8, 16, 24, 32, 40, 48, 64, 72, 80, 96):
            with self.subTest(batch_size=batch_size):
                natural, _ = self.sizer(
                    batch_size, 1, _HEADS, _KV_LORA_RANK, self.num_sms, max_seq_len
                )
                forced = floor_cutedsl_mla_decode_kv_splits(
                    split_sizer=self.sizer,
                    batch_size=batch_size,
                    num_heads=_HEADS,
                    kv_lora_rank=_KV_LORA_RANK,
                    num_sms=self.num_sms,
                    max_seq_len=max_seq_len,
                    min_kv_splits=_FLOOR,
                    workspace_bytes=_WORKSPACE_BYTES,
                )
                if natural >= _FLOOR:
                    self.assertIsNone(forced)
                    continue
                self.assertIsNotNone(forced)
                self.assertGreater(forced, natural)
                _, workspace = self.sizer(
                    batch_size,
                    1,
                    _HEADS,
                    _KV_LORA_RANK,
                    self.num_sms,
                    max_seq_len,
                    num_kv_splits=forced,
                )
                self.assertLessEqual(workspace, _WORKSPACE_BYTES)
                if batch_size <= 72:
                    self.assertEqual(forced, _FLOOR)

    def _skewed_batch(self, kv_dtype):
        """bs 40: one 900K request, 35 at 40K-200K, 4 padded rows of length 1,
        as DCP-8 rank-local lengths (also used as plain lengths without DCP,
        which keeps the KV cache small)."""
        gen = torch.Generator().manual_seed(0)
        real = 36
        global_lens = torch.cat(
            [
                torch.full((1,), 900_000),
                torch.randint(40_000, 200_000, (real - 1,), generator=gen),
                torch.ones(_B - real, dtype=torch.long),
            ]
        )
        local_lens = (global_lens + _DCP - 1) // _DCP
        pages = (local_lens + _PAGE_SIZE - 1) // _PAGE_SIZE
        num_pages = int(pages.sum()) + 1
        max_pages = (int(pages.max()) + 1) // 2 * 2
        dim = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM
        kv_cache = (torch.randn(num_pages, 1, _PAGE_SIZE, dim, device="cuda") * 0.5).to(
            kv_dtype
        )
        perm = torch.randperm(num_pages, generator=gen).cuda()
        block_tables = torch.zeros(_B, max_pages, dtype=torch.int32, device="cuda")
        offset = 0
        for i, n in enumerate(pages.tolist()):
            block_tables[i, :n] = perm[offset : offset + n].int()
            offset += n
        query = (torch.randn(_B, 1, _HEADS, dim, device="cuda") * 0.5).to(kv_dtype)
        return (
            query,
            kv_cache,
            block_tables,
            local_lens.int().cuda(),
            global_lens.int().cuda(),
            int(local_lens.max()),
            real,
        )

    def _check_against_default(self, kv_dtype, cp_world):
        import flashinfer

        query, kv_cache, block_tables, local_lens, global_lens, max_seq_len, real = (
            self._skewed_batch(kv_dtype)
        )
        enable_dcp = cp_world > 1
        dcp_kwargs = (
            dict(
                enable_dcp=True,
                cp_world=cp_world,
                cp_rank=0,
                causal_seqlens_kv_global=global_lens,
                return_lse=True,
            )
            if enable_dcp
            else dict(return_lse=True)
        )
        want_out, want_lse = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace,
            qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
            kv_lora_rank=_KV_LORA_RANK,
            qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
            block_tables=block_tables,
            seq_lens=local_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=_SCALING,
            backend="cute-dsl",
            **dcp_kwargs,
        )
        want_out, want_lse = want_out.clone(), want_lse.clone()

        natural, _ = self.sizer(_B, 1, _HEADS, _KV_LORA_RANK, self.num_sms, max_seq_len)
        self.assertEqual(natural, 1)
        num_kv_splits = floor_cutedsl_mla_decode_kv_splits(
            split_sizer=self.sizer,
            batch_size=_B,
            num_heads=_HEADS,
            kv_lora_rank=_KV_LORA_RANK,
            num_sms=self.num_sms,
            max_seq_len=max_seq_len,
            min_kv_splits=_FLOOR,
            workspace_bytes=_WORKSPACE_BYTES,
        )
        self.assertEqual(num_kv_splits, _FLOOR)
        got_out, got_lse = self._backend(kv_dtype)._run_decode_kernel_with_kv_splits(
            query,
            kv_cache,
            block_tables,
            local_lens,
            max_seq_len,
            SimpleNamespace(scaling=_SCALING, k_scale_float=None),
            num_kv_splits,
            causal_seqs=global_lens if enable_dcp else None,
            cp_world=cp_world,
            cp_rank=0,
            return_lse=True,
        )
        torch.cuda.synchronize()

        self.assertEqual(got_out.dtype, want_out.dtype)
        self.assertEqual(got_out.shape, want_out.shape)
        self.assertEqual(got_lse.shape, want_lse.shape)
        got_out = got_out.float().view(_B, _HEADS, -1)[:real]
        want_out = want_out.float().view(_B, _HEADS, -1)[:real]
        got_lse = got_lse.float().view(_B, _HEADS)[:real]
        want_lse = want_lse.float().view(_B, _HEADS)[:real]
        self.assertTrue(torch.isfinite(got_out).all() and torch.isfinite(got_lse).all())
        out_diff = (got_out - want_out).abs().max().item()
        lse_diff = (got_lse - want_lse).abs().max().item()
        self.assertLess(out_diff, 4e-3, f"out max |diff| {out_diff}")
        self.assertLess(lse_diff, 1e-4, f"lse max |diff| {lse_diff}")

    def test_dcp_fp8_decode_matches_default_split(self):
        # Kimi K3 serving shape: FP8 KV, DCP 8, rank 0.
        self._check_against_default(torch.float8_e4m3fn, _DCP)

    def test_bf16_decode_matches_default_split(self):
        self._check_against_default(torch.bfloat16, 1)


if __name__ == "__main__":
    unittest.main()
