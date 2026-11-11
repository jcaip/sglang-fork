# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla import chunk_delta_h_scan as seg_scan
from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.kernels.ops.attention.fla.op import exp, exp2, safe_exp
from sglang.kernels.ops.attention.fla.utils import (
    autotune_cache_kwargs,
    is_nvidia,
    is_nvidia_hopper,
)
from sglang.srt.environ import envs

NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]
CHUNK_SIZE = 64
GDN_CHUNK_H_BV = int(os.getenv("SGLANG_GDN_CHUNK_H_BV", "32"))
GDN_CHUNK_H_NUM_WARPS = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_WARPS", "4"))
GDN_CHUNK_H_NUM_STAGES = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_STAGES", "2"))


@triton.autotune(
    # Single hardcoded config. The kernel writes ht (final state) back into
    # initial_state in-place; with multiple configs, triton's autotune benchmark
    # phase invokes the kernel many times for timing and corrupts the cache pool,
    # producing silently wrong output on the first user request. Restoring via
    # `restore_value=["initial_state"]` works for unit tests but OOMs on
    # production-scale models (e.g. Kimi-Linear-48B at default mem_fraction)
    # because cloning the cache pool for each benchmark exceeds available memory.
    # NT_BUCKET is kept in the autotune key for forward-compatibility (allows
    # future per-bucket configs once the kernel is refactored to write final
    # state to a separate output buffer). The env knobs keep this single-config
    # property while allowing model/hardware-local validation of the selected
    # tile without corrupting the state pool through multi-config autotune.
    configs=[
        triton.Config(
            {"BV": GDN_CHUNK_H_BV},
            num_warps=GDN_CHUNK_H_NUM_WARPS,
            num_stages=GDN_CHUNK_H_NUM_STAGES,
        )
    ],
    key=["H", "K", "V", "BT", "USE_GK", "NT_BUCKET"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    initial_state,
    initial_state_indices,
    stride_init_state,
    cu_seqlens,
    chunk_offsets,
    track_state,
    track_chunk_idx,
    stride_track_state,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_UPDATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_BUCKET: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRACK_STATE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, BK]
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    # Slot stride comes from the caller (initial_state.stride(0)): the state pool
    # may be an envelope-strided view (page-major / unified memory), where the
    # per-slot pitch spans ALL layers' state, not H*V*K. int64: envelope pitches
    # overflow an int32 index product.
    index = tl.load(initial_state_indices + i_n).to(tl.int64)
    # Padded rows carry the -1 sentinel; the decode kernel guards on it
    # (fused_recurrent.py), the chunked extend path did not.
    valid_state = index >= 0
    h0 = initial_state + index * stride_init_state
    ht = initial_state + index * stride_init_state
    if USE_INITIAL_STATE:
        h0 = h0 + i_h * V * K
    if INPLACE_UPDATE:
        ht = ht + i_h * V * K

    if TRACK_STATE:
        i_track = tl.load(track_chunk_idx + i_n).to(tl.int32)
        p_track_base = track_state + (i_n * stride_track_state + i_h * V * K).to(
            tl.int64
        )
    else:
        i_track = -1
        p_track_base = track_state

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    m_h1 = m_v[:, None] & m_k1[None, :]
    m_h2 = m_v[:, None] & m_k2[None, :]
    m_h3 = m_v[:, None] & m_k3[None, :]
    m_h4 = m_v[:, None] & m_k4[None, :]

    # load initial state
    if USE_INITIAL_STATE and valid_state:
        p_h0_1 = h0 + o_v[:, None] * K + o_k1[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h1, other=0.0).to(tl.float32)
        if K > 64:
            p_h0_2 = h0 + o_v[:, None] * K + o_k2[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h2, other=0.0).to(tl.float32)
        if K > 128:
            p_h0_3 = h0 + o_v[:, None] * K + o_k3[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_h3, other=0.0).to(tl.float32)
        if K > 192:
            p_h0_4 = h0 + o_v[:, None] * K + o_k4[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_h4, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        o_t = (i_t * BT + tl.arange(0, BT)).to(tl.int64)
        m_t = o_t < T
        p_h1 = h + i_t.to(tl.int64) * stride_h + o_v[:, None] * K + o_k1[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_h2 = h + i_t.to(tl.int64) * stride_h + o_v[:, None] * K + o_k2[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_h3 = h + i_t.to(tl.int64) * stride_h + o_v[:, None] * K + o_k3[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_h4 = h + i_t.to(tl.int64) * stride_h + o_v[:, None] * K + o_k4[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

        if TRACK_STATE and i_t == i_track:
            p_t1 = p_track_base + o_v[:, None] * K + o_k1[None, :]
            tl.store(p_t1, b_h1, mask=m_h1)
            if K > 64:
                p_t2 = p_track_base + o_v[:, None] * K + o_k2[None, :]
                tl.store(p_t2, b_h2, mask=m_h2)
            if K > 128:
                p_t3 = p_track_base + o_v[:, None] * K + o_k3[None, :]
                tl.store(p_t3, b_h3, mask=m_h3)
            if K > 192:
                p_t4 = p_track_base + o_v[:, None] * K + o_k4[None, :]
                tl.store(p_t4, b_h4, mask=m_h4)

        p_w = w + o_t[:, None] * stride_w + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * stride_w + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = w + o_t[:, None] * stride_w + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = w + o_t[:, None] * stride_w + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        m_tv = m_t[:, None] & m_v[None, :]
        p_v = v + o_t[:, None] * stride_v + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tv, other=0.0) - b_v

        if SAVE_NEW_VALUE:
            p_v = v_new + o_t[:, None] * stride_v + o_v[None, :]
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_tv)

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos * H + i_h + o_t * H
            b_g = tl.load(p_g, mask=m_t, other=0.0)
            b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    # epilogue
    if INPLACE_UPDATE and valid_state:
        p_ht = ht + o_v[:, None] * K + o_k1[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_ht = ht + o_v[:, None] * K + o_k2[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_ht = ht + o_v[:, None] * K + o_k3[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_ht = ht + o_v[:, None] * K + o_k4[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_h4)


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    use_exp2: bool = False,
    inplace_update: bool = True,
    track_state: Optional[torch.Tensor] = None,
    track_chunk_idx: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert not (use_exp2 and g is not None), (
        "use_exp2 covers only the per-channel gk path; scalar g stays natural-exp"
    )
    assert (track_state is None) == (track_chunk_idx is None), (
        "track_state and track_chunk_idx must be passed together"
    )
    if track_state is not None:
        # The caller rounds once to the pool dtype; a narrower buffer would
        # silently double-round the snapshot.
        assert track_state.dtype == torch.float32, (
            f"track_state must be fp32, got {track_state.dtype}"
        )
    if (
        is_nvidia
        and envs.SGLANG_OPT_KDA_CHUNK_H_SEGMENT_SCAN.get()
        and seg_scan.segment_scan_eligible(
            k=k,
            u=u,
            g=g,
            gk=gk,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            use_exp2=use_exp2,
            inplace_update=inplace_update,
            chunk_offsets=chunk_offsets,
        )
    ):
        return _chunk_gated_delta_rule_fwd_h_segment_scan(
            k=k,
            w=w,
            u=u,
            gk=gk,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
        )
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = CHUNK_SIZE

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        # A caller-built chunk_offsets (CUDA-graph capture) must not go through
        # prepare_chunk_offsets: its tensor_cache would replay capture-time values.
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, V, K)

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        # Envelope-strided state pools (page-major / unified memory) have a
        # per-slot pitch != H*V*K; contiguous pools pass exactly H*V*K.
        stride_init_state=(initial_state.stride(0) if initial_state is not None else 0),
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
        stride_track_state=(track_state.stride(0) if track_state is not None else 0),
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_INITIAL_STATE=initial_state is not None,
        INPLACE_UPDATE=inplace_update,
        SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None,
        NT_BUCKET=(0 if NT <= 32 else (1 if NT <= 128 else 2)),
        USE_EXP2=use_exp2,
        TRACK_STATE=track_state is not None,
    )
    return h, v_new


def _chunk_gated_delta_rule_fwd_h_segment_scan(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    track_state: Optional[torch.Tensor],
    track_chunk_idx: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """chunk_gated_delta_rule_fwd_h for one long sequence, as a segment scan.

    Same outputs and side effects as the sequential path for the inputs
    `chunk_delta_h_scan.segment_scan_eligible` accepts: h and v_new, the final
    state written in place to the pool slot (no write for a -1 slot), and the
    track_state snapshot. See chunk_delta_h_scan.py for the three passes.
    """
    _, T, Hg, K = k.shape
    H, V = u.shape[-2], u.shape[-1]
    BT = CHUNK_SIZE
    NT = triton.cdiv(T, BT)
    seg_chunks, S = seg_scan.segment_scan_plan(T)
    device = k.device

    # Pass 1: each segment's affine transition [L_s; A_s] (the last segment's
    # is never needed).
    transition = torch.empty(
        max(S - 1, 1), H, V + K, K, device=device, dtype=torch.float32
    )
    if S > 1:
        seg_scan.segment_transition_kernel[
            (triton.cdiv(V + K, seg_scan.PASS1_BV), (S - 1) * H)
        ](
            k,
            u,
            w,
            gk,
            transition,
            seg_chunks * BT,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            BV=seg_scan.PASS1_BV,
            num_warps=4,
            num_stages=2,
        )

    # Pass 2: chain the segment entry states from the pool slot.
    entry = torch.empty(S, H, V, K, device=device, dtype=torch.float32)
    seg_scan.segment_entry_state_kernel[(triton.cdiv(V, seg_scan.PASS2_BV), H)](
        initial_state,
        initial_state_indices,
        initial_state.stride(0),
        transition,
        entry,
        S,
        H=H,
        K=K,
        V=V,
        BV=seg_scan.PASS2_BV,
        KB=seg_scan.PASS2_KB,
        num_warps=4,
    )

    # Pass 3: the sequential kernel over the segments as varlen sequences,
    # starting from `entry`. It writes each segment's final state back into
    # entry[s] in place, so entry[S - 1] ends as the sequence's final state.
    seg_bounds = (
        torch.arange(S + 1, device=device, dtype=torch.int32) * (seg_chunks * BT)
    ).clamp_max_(T)
    seg_ids = torch.arange(S, device=device, dtype=torch.int32)
    h = k.new_empty(1, NT, H, V, K)
    v_new = torch.empty_like(u)
    seg_track = seg_track_idx = None
    if track_state is not None:
        # Every segment gets the snapshot index relative to its own start; only
        # the segment holding the chunk matches and writes seg_track[s].
        seg_track = torch.empty(S, H, V, K, device=device, dtype=torch.float32)
        seg_track_idx = (
            track_chunk_idx[:1].to(torch.int32) - seg_ids * seg_chunks
        ).contiguous()

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), S * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=None,
        gk=gk,
        h=h,
        initial_state=entry,
        initial_state_indices=seg_ids,
        stride_init_state=entry.stride(0),
        cu_seqlens=seg_bounds,
        chunk_offsets=seg_bounds // BT,
        track_state=seg_track,
        track_chunk_idx=seg_track_idx,
        stride_track_state=(seg_track.stride(0) if seg_track is not None else 0),
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_G=False,
        USE_GK=True,
        USE_INITIAL_STATE=True,
        INPLACE_UPDATE=True,
        SAVE_NEW_VALUE=True,
        IS_VARLEN=True,
        NT_BUCKET=(0 if seg_chunks <= 32 else (1 if seg_chunks <= 128 else 2)),
        USE_EXP2=True,
        TRACK_STATE=seg_track is not None,
    )

    seg_scan.segment_final_state_writeback_kernel[(V // seg_scan.WRITEBACK_BV, H)](
        initial_state,
        initial_state_indices,
        initial_state.stride(0),
        entry[S - 1],
        H=H,
        K=K,
        V=V,
        BV=seg_scan.WRITEBACK_BV,
        num_warps=4,
    )

    if track_state is not None:
        # Like the sequential kernel, an index outside [0, NT) leaves the
        # caller's buffer untouched.
        track_idx = track_chunk_idx[:1].to(torch.int64)
        valid = (track_idx >= 0) & (track_idx < NT)
        snapshot = seg_track.index_select(0, (track_idx // seg_chunks).clamp(0, S - 1))
        track_state[:1].copy_(
            torch.where(valid.view(1, 1, 1, 1), snapshot, track_state[:1])
        )
    return h, v_new
