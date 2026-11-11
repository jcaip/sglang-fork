"""Long single-sequence KDA prefill state pass as a two-level segment scan.

`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` walks the T/64 chunks of a
sequence one after another on a (V/BV, N*H) grid. For one long sequence that is
a few dozen CTAs doing hundreds of dependent steps: Kimi K3 at TP8 (H=12,
K=V=128) launches 48 CTAs and takes 512 steps for a 32K prefill chunk.

Per 64-token chunk t, with the state S [V, K] in fp32:

    h[t]      = S_t
    v_new_t   = u_t - w_t S_t^T
    S_{t+1}   = S_t diag(exp2(gk_last_t)) + v_new_t^T k_t
              = S_t M_t + u_t^T k_t,    M_t = diag(exp2(gk_last_t)) - w_t^T k_t

so the update is affine in S, and a run of chunks maps S_in -> S_in A + L. The
scan splits the chunks into at most SEGMENTS segments and runs:

  pass 1 (parallel over segments 0..S-2): the upstream recurrence on an
         augmented [V+K, K] state that starts at [0; I] and reads the value
         [u, 0]. Rows < V end at L_s, rows >= V end at A_s. No h / v_new stores.
  pass 2 (one CTA column per head, loop over segments): S_in[0] = pool slot,
         S_in[s+1] = S_in[s] A_s + L_s with fp32 IEEE dots.
  pass 3 (parallel over segments): the regular kernel, with the segments as
         varlen sequences that start from S_in[s]. Segment bounds are multiples
         of 64, so each segment's chunk offset is its global chunk index and h
         lands where the sequential kernel would put it.

The result is close to the sequential kernel but not bit-identical: that kernel
rounds S and v_new to bf16 inside its dots, and the scan reaches each S_t by a
different product order. Against an fp64 recurrence its error matches the
sequential kernel's own.
"""

import triton
import triton.language as tl

# Segments per sequence. Eight is enough to fill a B300 for K3 at TP8 (pass 1
# launches (V+K)/32 * 7 * H CTAs) while keeping pass 2's serial chain short.
SEGMENTS = 8
# Shortest sequence, in 64-token chunks, the scan takes; below this the
# sequential kernel's own grid is cheaper than the three extra passes.
MIN_CHUNKS = 128
CHUNK_SIZE = 64
PASS1_BV = 32
PASS2_BV = 16
PASS2_KB = 32
WRITEBACK_BV = 16


def segment_scan_plan(num_tokens: int):
    """(chunks per segment, number of segments) for a sequence of num_tokens.

    Every segment is non-empty, all but the last hold exactly the first value
    chunks, and the segments cover every chunk once.
    """
    num_chunks = triton.cdiv(num_tokens, CHUNK_SIZE)
    seg_chunks = triton.cdiv(num_chunks, SEGMENTS)
    return seg_chunks, triton.cdiv(num_chunks, seg_chunks)


def segment_scan_eligible(
    k,
    u,
    g,
    gk,
    initial_state,
    initial_state_indices,
    save_new_value,
    cu_seqlens,
    use_exp2,
    inplace_update,
    chunk_offsets,
) -> bool:
    """Whether the segment scan can stand in for the sequential kernel.

    Covered: one sequence (batch 1, at most one varlen row), K == 128, the
    per-channel gk gate with exp2, a state pool updated in place, v_new kept,
    and at least MIN_CHUNKS chunks. A caller-built chunk_offsets is the
    prefill CUDA-graph path, whose rows are padded past the real length and
    whose cu_seqlens are only known at replay; that path stays sequential.
    """
    if not (k.is_cuda and k.dim() == 4 and u.dim() == 4):
        return False
    if k.shape[0] != 1 or k.shape[-1] != 128:
        return False
    if g is not None or gk is None or not use_exp2:
        return False
    if initial_state is None or initial_state_indices is None:
        return False
    if not (inplace_update and save_new_value) or chunk_offsets is not None:
        return False
    if cu_seqlens is not None and len(cu_seqlens) != 2:
        return False
    H, V, K = u.shape[-2], u.shape[-1], k.shape[-1]
    if V % WRITEBACK_BV or H % k.shape[2]:
        return False
    # The scan passes address one slot as a dense [H, V, K] block; the slot
    # pitch itself may be envelope-strided.
    if initial_state.dim() != 4 or tuple(initial_state.stride()[1:]) != (
        V * K,
        K,
        1,
    ):
        return False
    return triton.cdiv(k.shape[1], CHUNK_SIZE) >= MIN_CHUNKS


@triton.jit
def segment_transition_kernel(
    k,
    v,
    w,
    gk,
    transition,
    T_SEG,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    # Pass 1. The sequential recurrence (K=128: two 64-column halves) on rows
    # [i_v*BV, i_v*BV+BV) of the [V+K, K] augmented state. Rows >= V start at
    # the identity and read no value (the (T, V) block pointer zero-fills
    # columns >= V). Output: transition[s] = [L_s; A_s].
    i_v, i_sh = tl.program_id(0), tl.program_id(1)
    i_s, i_h = i_sh // H, i_sh % H
    bos = i_s * T_SEG
    r = i_v * BV + tl.arange(0, BV) - V
    o_k = tl.arange(0, 64)
    b_h1 = (r[:, None] == o_k[None, :]).to(tl.float32)
    b_h2 = (r[:, None] == (o_k + 64)[None, :]).to(tl.float32)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    NT = T_SEG // BT
    for i_t in range(NT):
        p_w = tl.make_block_ptr(
            w, (T_SEG, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        p_w = tl.make_block_ptr(
            w, (T_SEG, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T_SEG, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v
        last = (bos + i_t * BT + BT - 1) * H * K + i_h * K
        b_h1 *= tl.math.exp2(tl.load(gk + last + o_k))[None, :]
        b_h2 *= tl.math.exp2(tl.load(gk + last + 64 + o_k))[None, :]
        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(
            k, (K, T_SEG), (1, Hg * K), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_h1 += tl.trans(tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_v))
        p_k = tl.make_block_ptr(
            k, (K, T_SEG), (1, Hg * K), (64, i_t * BT), (64, BT), (0, 1)
        )
        b_h2 += tl.trans(tl.dot(tl.load(p_k, boundary_check=(0, 1)), b_v))
    transition += ((i_s * H + i_h) * (V + K) * K).to(tl.int64)
    p = tl.make_block_ptr(
        transition, (V + K, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
    )
    tl.store(p, b_h1, boundary_check=(0, 1))
    p = tl.make_block_ptr(
        transition, (V + K, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
    )
    tl.store(p, b_h2, boundary_check=(0, 1))


@triton.jit
def segment_entry_state_kernel(
    pool,
    pool_idx,
    stride_pool,
    transition,
    entry,
    S,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    KB: tl.constexpr,
):
    # Pass 2. entry[0] = pool slot (zeros for a -1 slot);
    # entry[s+1] = entry[s] A_s + L_s, over K in KB-wide blocks (a whole
    # [K, K] fp32 A tile in registers spills). Each step re-reads entry[s] from
    # the rows this CTA just stored, hence the barrier.
    i_v, i_h = tl.program_id(0), tl.program_id(1)
    o_r = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    o_b = tl.arange(0, KB)
    idx = tl.load(pool_idx).to(tl.int64)
    b_s = tl.load(
        pool + idx * stride_pool + i_h * V * K + o_r[:, None] * K + o_k[None, :],
        mask=(idx >= 0) & (o_r[:, None] < V),
        other=0.0,
    ).to(tl.float32)
    for i_s in range(S):
        base = (i_s * H + i_h).to(tl.int64)
        p_s = entry + base * V * K
        tl.store(p_s + o_r[:, None] * K + o_k[None, :], b_s, mask=o_r[:, None] < V)
        if i_s < S - 1:
            tl.debug_barrier()
            p_tr = transition + base * (V + K) * K
            b_s = tl.load(
                p_tr + o_r[:, None] * K + o_k[None, :],
                mask=o_r[:, None] < V,
                other=0.0,
            )
            for kb in tl.static_range(0, K, KB):
                b_x = tl.load(
                    p_s + o_r[:, None] * K + (kb + o_b)[None, :],
                    mask=o_r[:, None] < V,
                    other=0.0,
                )
                b_a = tl.load(p_tr + (V + kb + o_b)[:, None] * K + o_k[None, :])
                b_s += tl.dot(b_x, b_a, input_precision="ieee")


@triton.jit
def segment_final_state_writeback_kernel(
    pool,
    pool_idx,
    stride_pool,
    final_state,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    # The last segment's final state goes to the pool slot; a -1 slot is not
    # written, as in the sequential kernel.
    i_v, i_h = tl.program_id(0), tl.program_id(1)
    idx = tl.load(pool_idx).to(tl.int64)
    if idx >= 0:
        o = (
            i_h * V * K
            + (i_v * BV + tl.arange(0, BV))[:, None] * K
            + tl.arange(0, K)[None, :]
        )
        b = tl.load(final_state + o)
        tl.store(pool + idx * stride_pool + o, b.to(pool.dtype.element_ty))
