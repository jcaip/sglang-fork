"""KDA prefill state pass: segment scan vs the sequential kernel (GPU).

With SGLANG_OPT_KDA_CHUNK_H_SEGMENT_SCAN=1, chunk_gated_delta_rule_fwd_h runs a
long single sequence as a two-level segment scan. It is not bit-identical to
the sequential kernel (different product order; the sequential kernel rounds S
and v_new to bf16 inside its dots), so the contract is: against an fp64
recurrence, h, v_new and the final pool state are no worse than the sequential
kernel's own error; the track_state snapshot matches; other pool slots are
untouched; a -1 slot is never written; ineligible calls keep the sequential
kernel.

    python3 test/registered/kernels/ops/attention/test_kda_chunk_h_segment_scan.py
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla import chunk_delta_h
from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Kimi K3 KDA at TP8.
H, K, V = 12, 128, 128
POOL = 4


def _inputs(T, slot, track=None, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda")

    k = torch.nn.functional.normalize(randn(1, T, H, K), dim=-1).to(torch.bfloat16)
    w = (randn(1, T, H, K) * 0.05).to(torch.bfloat16)
    u = (randn(1, T, H, V) * 0.5).to(torch.bfloat16)
    # Per-channel log2 decay, cumulative within each 64-token chunk (what
    # chunk_kda hands this function).
    gk = -torch.rand(1, T, H, K, generator=g, device="cuda") * 0.1
    gk = torch.cat([c.cumsum(1) for c in gk.split(64, dim=1)], dim=1).contiguous()
    kw = dict(
        k=k,
        w=w,
        u=u,
        gk=gk,
        initial_state=randn(POOL, H, V, K) * 0.1,
        initial_state_indices=torch.tensor([slot], device="cuda", dtype=torch.int32),
        cu_seqlens=torch.tensor([0, T], device="cuda", dtype=torch.int32),
        use_exp2=True,
    )
    if track is not None:
        kw["track_state"] = torch.full(
            (1, H, V, K), 7.0, device="cuda", dtype=torch.float32
        )
        kw["track_chunk_idx"] = torch.tensor([track], device="cuda", dtype=torch.int32)
    return kw


def _clone(kw):
    return {a: (b.clone() if isinstance(b, torch.Tensor) else b) for a, b in kw.items()}


def _fp64_reference(kw, slot):
    k, w, u, gk = kw["k"], kw["w"], kw["u"], kw["gk"]
    T = k.shape[1]
    S = (
        kw["initial_state"][slot].double()
        if slot >= 0
        else torch.zeros(H, V, K, device="cuda", dtype=torch.float64)
    )
    hs, vs = [], []
    for t in range((T + 63) // 64):
        sl = slice(t * 64, min(T, t * 64 + 64))
        kd, wd, ud = (x[0, sl].double().transpose(0, 1) for x in (k, w, u))
        hs.append(S)
        vn = ud - wd @ S.transpose(1, 2)
        vs.append(vn.transpose(0, 1))
        S = (
            S * torch.exp2(gk[0, sl.stop - 1].double())[:, None, :]
            + vn.transpose(1, 2) @ kd
        )
    return torch.stack(hs)[None], torch.cat(vs)[None], S


def _rel_err(a, ref):
    d = a.double() - ref.double()
    return (d.norm() / ref.double().norm().clamp_min(1e-30)).item()


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestKDAChunkHSegmentScan(CustomTestCase):
    def _run(self, kw, enable):
        with envs.SGLANG_OPT_KDA_CHUNK_H_SEGMENT_SCAN.override(enable):
            return chunk_gated_delta_rule_fwd_h(**kw)

    def test_scan_matches_sequential_error(self):
        # (T, pool slot, track chunk): segment counts 8 and 7, a ragged tail,
        # a snapshot in the last chunk, and a padded (-1) slot.
        cases = [
            (8192, 2, None),
            (16384, 1, 100),
            (32768, 3, 300),
            (32768 - 37, 0, 511),
            (20000, -1, 5),
        ]
        for T, slot, track in cases:
            with self.subTest(T=T, slot=slot, track=track):
                kw = _inputs(T, slot, track)
                seq, scan = _clone(kw), _clone(kw)
                with patch.object(
                    chunk_delta_h,
                    "_chunk_gated_delta_rule_fwd_h_segment_scan",
                    wraps=chunk_delta_h._chunk_gated_delta_rule_fwd_h_segment_scan,
                ) as scan_path:
                    h_seq, v_seq = self._run(seq, False)
                    self.assertEqual(scan_path.call_count, 0)
                    h_scan, v_scan = self._run(scan, True)
                    self.assertEqual(scan_path.call_count, 1)

                h64, v64, s64 = _fp64_reference(kw, slot)
                for name, out, ref, exact in [
                    ("h", h_scan, h_seq, h64),
                    ("v_new", v_scan, v_seq, v64),
                ]:
                    # 10% slack for the different product order.
                    self.assertLessEqual(
                        _rel_err(out, exact), 1.1 * _rel_err(ref, exact) + 1e-6, name
                    )

                others = [i for i in range(POOL) if i != slot]
                self.assertTrue(
                    torch.equal(
                        scan["initial_state"][others], seq["initial_state"][others]
                    )
                )
                if slot >= 0:
                    self.assertLessEqual(
                        _rel_err(scan["initial_state"][slot], s64),
                        1.1 * _rel_err(seq["initial_state"][slot], s64) + 1e-6,
                    )
                else:
                    self.assertTrue(
                        torch.equal(scan["initial_state"], kw["initial_state"])
                    )

                if track is not None:
                    self.assertLess(
                        _rel_err(scan["track_state"], seq["track_state"]), 1e-2
                    )
                    # The snapshot is the state the scan stored as h[track].
                    torch.testing.assert_close(
                        scan["track_state"][0],
                        h_scan[0, track].float(),
                        atol=1e-2,
                        rtol=0,
                    )

    def test_ineligible_calls_stay_sequential(self):
        two_seqs = _inputs(16384, 1)
        two_seqs["cu_seqlens"] = torch.tensor(
            [0, 8192, 16384], device="cuda", dtype=torch.int32
        )
        two_seqs["initial_state_indices"] = torch.tensor(
            [1, 2], device="cuda", dtype=torch.int32
        )
        graph_path = _inputs(16384, 1)
        graph_path["chunk_offsets"] = torch.tensor(
            [0, 256], device="cuda", dtype=torch.int32
        )
        cases = {
            "short": _inputs(8192 - 64, 1),
            "two_seqs": two_seqs,
            "graph_path": graph_path,
        }
        for name, kw in cases.items():
            with (
                self.subTest(name),
                patch.object(
                    chunk_delta_h, "_chunk_gated_delta_rule_fwd_h_segment_scan"
                ) as scan_path,
            ):
                on, off = _clone(kw), _clone(kw)
                h_on, v_on = self._run(on, True)
                h_off, v_off = self._run(off, False)
                scan_path.assert_not_called()
                self.assertTrue(torch.equal(h_on, h_off))
                self.assertTrue(torch.equal(v_on, v_off))
                self.assertTrue(torch.equal(on["initial_state"], off["initial_state"]))


if __name__ == "__main__":
    unittest.main()
