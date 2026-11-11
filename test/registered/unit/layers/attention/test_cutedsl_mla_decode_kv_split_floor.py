"""SGLANG_CUTEDSL_MLA_DECODE_MIN_KV_SPLITS: the cutedsl_mla decode split floor.

FlashInfer's CuTe DSL MLA decode picks one split count per launch from
occupancy only (about ``num_sms // (2 * batch)``), so past bs ~37 on a 148-SM
GPU every request runs on one CTA and the longest request in the batch sets the
launch time. ``floor_cutedsl_mla_decode_kv_splits`` raises that count to the
floor, or the largest count below it whose split-KV workspace fits the backend's
buffer, and must never lower FlashInfer's choice.

The fake sizer mirrors FlashInfer 0.7.3's
``_get_split_kv_and_workspace_size`` for one 128-row Q tile per request:
``split = min(32, max_active_blocks // (2 * B))`` normalized to the non-empty
128-token K tiles of ``max_seq_len``, an explicit ``num_kv_splits`` taken as is,
and ``B * 128 * split * (kv_lora_rank + 1) * 4`` workspace bytes (0 at split 1).

CPU-only.

    python -m pytest test/registered/unit/layers/attention/test_cutedsl_mla_decode_kv_split_floor.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import cutedsl_mla_backend
from sglang.srt.layers.attention.cutedsl_mla_backend import (
    CuteDslMLABackend,
    floor_cutedsl_mla_decode_kv_splits,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Kimi K3 per-GPU decode shape on a 148-SM B300; 150 MB is the cute-dsl
# workspace TRTLLMMLABackend allocates (DEFAULT_WORKSPACE_SIZE_MB).
HEADS, KV_LORA_RANK, SMS = 12, 512, 148
WORKSPACE_BYTES = 150 * 1024 * 1024
LONG_CONTEXT = 100_000


def _ceil_div(a, b):
    return -(-a // b)


def _flashinfer_sizer(
    B, q_len, H, kv_lora_rank, max_active_blocks, max_seq_len, num_kv_splits=None
):
    assert q_len == 1 and H <= 128
    if num_kv_splits is None:
        split = max(1, min(32, max_active_blocks // (2 * B)))
        k_tiles = _ceil_div(max_seq_len, 128)
        split = _ceil_div(k_tiles, _ceil_div(k_tiles, split))
    else:
        split = num_kv_splits
    workspace = 0 if split == 1 else B * 128 * split * (kv_lora_rank + 1) * 4
    return split, workspace


def _floor(batch_size, min_kv_splits=8, max_seq_len=LONG_CONTEXT, sizer=None):
    return floor_cutedsl_mla_decode_kv_splits(
        split_sizer=sizer or _flashinfer_sizer,
        batch_size=batch_size,
        num_heads=HEADS,
        kv_lora_rank=KV_LORA_RANK,
        num_sms=SMS,
        max_seq_len=max_seq_len,
        min_kv_splits=min_kv_splits,
        workspace_bytes=WORKSPACE_BYTES,
    )


def _workspace(batch_size, splits):
    return _flashinfer_sizer(
        batch_size, 1, HEADS, KV_LORA_RANK, SMS, LONG_CONTEXT, splits
    )[1]


class TestCuteDslMLADecodeKVSplitFloor(CustomTestCase):
    def test_raises_occupancy_split_to_the_floor(self):
        # FlashInfer gives 2-3 splits at bs 24-32 and 1 from bs 40.
        for batch_size in (24, 32, 40, 48, 64, 74):
            with self.subTest(batch_size=batch_size):
                natural = _flashinfer_sizer(
                    batch_size, 1, HEADS, KV_LORA_RANK, SMS, LONG_CONTEXT
                )[0]
                self.assertLess(natural, 8)
                self.assertEqual(_floor(batch_size), 8)
                self.assertLessEqual(_workspace(batch_size, 8), WORKSPACE_BYTES)

    def test_workspace_lowers_the_floor_to_what_fits(self):
        # 8 splits stop fitting 150 MB at bs 75; bs 96 fits 6 but not 7.
        self.assertGreater(_workspace(75, 8), WORKSPACE_BYTES)
        self.assertEqual(_floor(75), 7)
        self.assertEqual(_floor(96), 6)
        self.assertLessEqual(_workspace(96, 6), WORKSPACE_BYTES)
        self.assertGreater(_workspace(96, 7), WORKSPACE_BYTES)

    def test_keeps_flashinfer_choice_at_or_above_the_floor(self):
        # Small batches already split widely; the floor must not lower them.
        for batch_size in (1, 2, 4, 9):
            with self.subTest(batch_size=batch_size):
                self.assertIsNone(_floor(batch_size))
        self.assertIsNone(_floor(40, min_kv_splits=1))

    def test_no_raise_when_nothing_larger_fits(self):
        # At bs 512 not even 2 splits fit 150 MB: keep FlashInfer's 1 split.
        self.assertGreater(_workspace(512, 2), WORKSPACE_BYTES)
        self.assertIsNone(_floor(512))

    def test_split_is_normalized_to_nonempty_k_tiles(self):
        # 300 tokens are 3 K tiles: no split count above 3 is non-empty.
        self.assertEqual(_floor(40, max_seq_len=300), 3)
        # 1000 tokens are 8 K tiles: a floor of 6 rounds to 4 two-tile splits.
        self.assertEqual(_floor(40, min_kv_splits=6, max_seq_len=1000), 4)
        # One K tile cannot split at all.
        self.assertIsNone(_floor(40, max_seq_len=100))

    def test_sizer_refusal_keeps_flashinfer_choice(self):
        def refusing_sizer(*args, num_kv_splits=None):
            if num_kv_splits is not None:
                raise ValueError("workspace exceeds int32 indexing")
            return _flashinfer_sizer(*args)

        self.assertIsNone(_floor(40, sizer=refusing_sizer))


class TestCuteDslMLABackendKVSplitGate(CustomTestCase):
    """Only a plain q_len-1 decode takes the explicit-split kernel call; every
    other launch keeps FlashInfer's own split and error paths."""

    def _backend(self, min_kv_splits=8):
        backend = CuteDslMLABackend.__new__(CuteDslMLABackend)
        backend._decode_min_kv_splits = min_kv_splits
        backend._cutedsl_mla_decode = SimpleNamespace(
            _get_split_kv_and_workspace_size=_flashinfer_sizer
        )
        backend._device_sm_count = SMS
        backend._decode_kv_splits_cache = {}
        backend.kv_lora_rank = KV_LORA_RANK
        backend.workspace_buffer = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8)
        return backend

    def _splits(self, backend, q_len=1, cp_world=1, dcp_enabled=False):
        query = torch.empty((40, q_len, HEADS, 1), dtype=torch.bfloat16)
        with patch.object(
            cutedsl_mla_backend,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=dcp_enabled),
        ):
            return backend._decode_kv_split_floor(query, LONG_CONTEXT, cp_world)

    def test_gate(self):
        backend = self._backend()
        self.assertEqual(self._splits(backend), 8)
        self.assertEqual(self._splits(backend, cp_world=8, dcp_enabled=True), 8)
        # Speculative verify / draft-extend (q_len > 1).
        self.assertIsNone(self._splits(backend, q_len=4))
        # A non-DCP launch under DCP keeps TRTLLMMLABackend's refusal checks.
        self.assertIsNone(self._splits(backend, dcp_enabled=True))
        # cute-dsl refuses skip-softmax; FlashInfer must still raise it.
        with envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.override(1.0):
            self.assertIsNone(self._splits(backend))
        # Off by default.
        self.assertIsNone(self._splits(self._backend(min_kv_splits=0)))


if __name__ == "__main__":
    unittest.main()
