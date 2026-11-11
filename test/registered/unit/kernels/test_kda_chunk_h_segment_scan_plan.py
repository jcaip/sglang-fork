"""KDA prefill segment scan: segment plan and eligibility gate (CPU).

The scan (fla/chunk_delta_h_scan.py) reruns the sequential state kernel over
its segments as varlen sequences and relies on two derived properties of
`segment_scan_plan`: every segment starts on a 64-token chunk boundary (so the
varlen chunk offset equals the global chunk index and h lands in place), and the
segments are non-empty and cover each chunk once (so the final state and the
track snapshot come from the right segment). `segment_scan_eligible` must turn
away every input the scan does not reproduce; in particular the prefill
CUDA-graph path (caller-built chunk_offsets), whose rows are padded past the
real sequence, must stay on the sequential kernel.

Both functions are pure shape arithmetic; they are loaded from source so the
test runs without Triton or a GPU.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/kernels/ops/attention/fla/chunk_delta_h_scan.py"
)


def _load():
    tree = ast.parse(_SOURCE.read_text())
    keep = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in ("segment_scan_plan", "segment_scan_eligible")
        )
        or (
            isinstance(node, ast.Assign)
            and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)
        )
    ]
    namespace = {"triton": SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d)}
    exec(
        compile(ast.Module(body=keep, type_ignores=[]), str(_SOURCE), "exec"), namespace
    )
    return namespace


_NS = _load()


class _Meta:
    """Shape/stride stand-in for a CUDA tensor."""

    def __init__(self, shape, stride=None):
        self.shape = tuple(shape)
        self.is_cuda = True
        if stride is None:
            stride, acc = [], 1
            for n in reversed(self.shape):
                stride.insert(0, acc)
                acc *= n
        self._stride = tuple(stride)

    def dim(self):
        return len(self.shape)

    def stride(self):
        return self._stride

    def __len__(self):
        return self.shape[0]


H, K, V = 12, 128, 128


def _k3_call(T, **overrides):
    kw = dict(
        k=_Meta((1, T, H, K)),
        u=_Meta((1, T, H, V)),
        g=None,
        gk=_Meta((1, T, H, K)),
        initial_state=_Meta((4, H, V, K)),
        initial_state_indices=_Meta((1,)),
        save_new_value=True,
        cu_seqlens=_Meta((2,)),
        use_exp2=True,
        inplace_update=True,
        chunk_offsets=None,
    )
    kw.update(overrides)
    return kw


class TestKDAChunkHSegmentScanPlan(CustomTestCase):
    def test_segments_cover_every_chunk_on_chunk_boundaries(self):
        plan = _NS["segment_scan_plan"]
        segments, chunk = _NS["SEGMENTS"], _NS["CHUNK_SIZE"]
        for T in list(range(8000, 8400, 37)) + [
            16384,
            20000,
            32768 - 37,
            32768,
            131071,
        ]:
            seg_chunks, S = plan(T)
            num_chunks = -(-T // chunk)
            # Bounds as the scan builds them: s * seg_chunks * 64, clamped to T.
            bounds = [min(s * seg_chunks * chunk, T) for s in range(S + 1)]
            with self.subTest(T=T):
                self.assertLessEqual(S, segments)
                self.assertEqual(bounds[0], 0)
                self.assertEqual(bounds[-1], T)
                self.assertTrue(all(b % chunk == 0 for b in bounds[:-1]))
                self.assertTrue(all(a < b for a, b in zip(bounds, bounds[1:])))
                # Pass 1 builds transitions for segments 0..S-2 as full segments.
                self.assertLessEqual((S - 1) * seg_chunks * chunk, T)
                # Every chunk's snapshot lives in segment t // seg_chunks at a
                # local index inside that segment.
                for t in range(num_chunks):
                    s = t // seg_chunks
                    self.assertLess(s, S)
                    self.assertLessEqual(bounds[s], t * chunk)
                    self.assertLess(t * chunk, bounds[s + 1])

    def test_eligibility(self):
        eligible = _NS["segment_scan_eligible"]
        min_tokens = _NS["MIN_CHUNKS"] * _NS["CHUNK_SIZE"]
        self.assertTrue(eligible(**_k3_call(min_tokens)))
        self.assertTrue(eligible(**_k3_call(32768, cu_seqlens=None)))
        # Envelope-strided pool (page-major / unified memory): any slot pitch.
        envelope = _Meta((4, H, V, K), stride=(7 * H * V * K, V * K, K, 1))
        self.assertTrue(eligible(**_k3_call(32768, initial_state=envelope)))

        T = 32768
        rejected = {
            "too_short": _k3_call(min_tokens - _NS["CHUNK_SIZE"]),
            "two_sequences": _k3_call(T, cu_seqlens=_Meta((3,))),
            "batch_2": _k3_call(
                T, k=_Meta((2, T, H, K)), u=_Meta((2, T, H, V)), cu_seqlens=None
            ),
            "cuda_graph_offsets": _k3_call(T, chunk_offsets=_Meta((2,))),
            "scalar_gate": _k3_call(T, g=_Meta((1, T, H))),
            "no_gate": _k3_call(T, gk=None),
            "natural_exp": _k3_call(T, use_exp2=False),
            "no_pool": _k3_call(T, initial_state=None),
            "no_indices": _k3_call(T, initial_state_indices=None),
            "out_of_place": _k3_call(T, inplace_update=False),
            "no_v_new": _k3_call(T, save_new_value=False),
            "head_dim_64": _k3_call(T, k=_Meta((1, T, H, 64)), gk=_Meta((1, T, H, 64))),
            "strided_slot": _k3_call(
                T,
                initial_state=_Meta(
                    (4, H, V, K), stride=(H * V * K * 2, V * K * 2, K, 1)
                ),
            ),
        }
        for name, kw in rejected.items():
            with self.subTest(name):
                self.assertFalse(eligible(**kw))


if __name__ == "__main__":
    unittest.main()
