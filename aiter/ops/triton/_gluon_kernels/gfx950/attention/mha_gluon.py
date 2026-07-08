##############################################################################
# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

import torch
import triton
from packaging.version import Version

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    _GLUON_IMPORTED = True
except Exception:  # pragma: no cover - depends on triton build / arch
    # Keep this module importable on builds without Gluon so the wrapper and
    # unit tests can query availability / feature support without a hard import
    # failure. The @jit kernels below are never callable in this mode; callers
    # guard on is_gluon_available() before dispatching to them.
    import triton.language as gl  # provides gl.constexpr for the signatures

    _GLUON_IMPORTED = False

    class _GluonFallback:
        @staticmethod
        def jit(fn=None, **_kwargs):
            def _wrap(f):
                return f

            return _wrap(fn) if fn is not None else _wrap

        @staticmethod
        def constexpr_function(fn=None, **_kwargs):
            def _wrap(f):
                return f

            return _wrap(fn) if fn is not None else _wrap

    gluon = _GluonFallback()

from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


# ---------------------------------------------------------------------------
# Gluon backend capability helpers
#
# This is a forward-only flash-attention kernel and does NOT implement the full
# feature set of the default Triton MHA backend. These helpers are the single
# source of truth for "can the Gluon backend serve this call?" and are shared by
# the mha.py wrapper (to reject unsupported calls) and the unit tests (to skip
# unsupported parametrizations).
# ---------------------------------------------------------------------------
_GLUON_SUPPORTED_ARCHS = ("gfx950",)
_TRITON_GE_36 = Version(triton.__version__) >= Version("3.6.0")


def is_gluon_available() -> bool:
    """True when this Gluon MHA forward kernel can actually run on this device."""
    if not (_GLUON_IMPORTED and _TRITON_GE_36):
        return False
    try:
        arch = get_arch() or ""
        return any(supported in arch for supported in _GLUON_SUPPORTED_ARCHS)
    except Exception:
        return False


def gluon_forward_unsupported_reason(
    *,
    head_dim=None,
    num_k_heads=None,
    dropout_p: float = 0.0,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    sink=None,
    return_lse: bool = False,
    return_attn_probs: bool = False,
    is_fp8: bool = False,
    has_positional_encoding: bool = False,
    block_table=None,
):
    """Reason (str) why the Gluon forward backend can't serve this config, else None.

    Shared by the mha.py wrapper (which turns a non-None reason into an error) and
    by the unit tests (which turn it into a ``pytest.skip``). Pass the feature
    flags and, optionally, ``head_dim`` / ``num_k_heads`` for the shape checks.
    """
    if not is_gluon_available():
        return (
            f"Gluon MHA backend requires one of {_GLUON_SUPPORTED_ARCHS} with "
            f"Triton>=3.6 (arch={get_arch()!r}, triton={triton.__version__})"
        )
    if dropout_p and dropout_p != 0.0:
        return "Gluon MHA backend does not support dropout"
    if tuple(window_size) != (-1, -1):
        return "Gluon MHA backend does not support sliding window attention"
    if bias is not None:
        return "Gluon MHA backend does not support attention bias"
    if alibi_slopes is not None:
        return "Gluon MHA backend does not support alibi slopes"
    if sink is not None:
        return "Gluon MHA backend does not support attention sink"
    if return_lse:
        return "Gluon MHA backend does not support returning LSE"
    if return_attn_probs:
        return "Gluon MHA backend does not support returning attention probabilities"
    if is_fp8:
        return "Gluon MHA backend does not support FP8"
    if has_positional_encoding:
        return "Gluon MHA backend does not support positional encoding"
    if block_table is not None:
        return "Gluon MHA backend does not support paged KV (block_table)"
    if head_dim is not None and num_k_heads is not None:
        # Known-bugged: a padded head_dim (padded up to a power of 2 >= 32)
        # combined with a KV sequence stride (num_k_heads * head_dim) that is not
        # a multiple of 16 elements fails to legalize the masked async
        # global->LDS copy on gfx950.
        padded_head = head_dim != max(triton.next_power_of_2(head_dim), 32)
        if padded_head and (num_k_heads * head_dim) % 16 != 0:
            return (
                "Gluon MHA backend: padded head_dim with a non-16-element-aligned "
                f"KV stride (head_dim={head_dim}, num_k_heads={num_k_heads}) is "
                "currently broken"
            )
    return None


@gluon.constexpr_function
def _make_kv_shared_layouts(
    head_dim_pow2, elem_bytes, k_width=8, non_k_dim=16, banks=64
):
    """Swizzled LDS layouts for the async K/V staging tiles."""
    bank_line_bytes = banks * 4
    bank_line_elems = bank_line_bytes // elem_bytes
    read_vec_bytes = min(k_width * elem_bytes, 16)
    num_threads_same_cycle = bank_line_bytes // read_vec_bytes
    per_phase = (bank_line_elems + head_dim_pow2 - 1) // head_dim_pow2
    swizzle_vec = k_width * max(1, per_phase // 2)
    max_phase = min(
        min(non_k_dim, num_threads_same_cycle) // per_phase,
        bank_line_elems // swizzle_vec,
    )
    k_shared = gl.SwizzledSharedLayout(swizzle_vec, per_phase, max_phase, order=[0, 1])
    v_shared = gl.SwizzledSharedLayout(swizzle_vec, per_phase, max_phase, order=[1, 0])
    return k_shared, v_shared


@gluon.jit
def _async_load_fn(smem, base, offsets):
    """Unmasked async global->LDS copy. Only for fully in-range tiles; masked
    tiles use _async_load_masked_fn. Caller owns commit_group / wait_group."""
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem, base, offsets)


@gluon.jit
def _async_load_masked_fn(
    smem, base, offsets, offset_first, offset_second, boundary_first, boundary_second
):
    """Masked async global->LDS copy. Masked lanes read other=0 so out-of-range
    key columns / head-dim padding contribute 0 to the matmuls. Each axis is
    masked only when its offset tensor is not None. Caller owns commit/wait."""
    ptrs = base + offsets
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (
            offset_second[None, :] < boundary_second
        )
        gl.amd.cdna4.async_copy.global_load_to_shared(smem, ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        gl.amd.cdna4.async_copy.global_load_to_shared(smem, ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        gl.amd.cdna4.async_copy.global_load_to_shared(smem, ptrs, mask=mask, other=0.0)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(smem, ptrs)


@gluon.jit
def _issue_kv_copy(
    smemK_buf,
    smemV_buf,
    k_base,
    v_base,
    k_offsets,
    v_offsets,
    copy_start_n,
    seqlen_k,
    K_LOAD_LAYOUT: gl.constexpr,
    V_LOAD_LAYOUT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    MASK_STEPS: gl.constexpr,
    PADDED_HEAD: gl.constexpr,
):
    """Issue the async global->LDS copies of one K and one V key block into the
    given buffer slots. K is staged as [BLOCK_DMODEL_POW2, BLOCK_N], V as
    [BLOCK_N, BLOCK_DMODEL_POW2]. Copy_start_n is the tile's key-block
    start (used for the n-axis mask). Caller owns commit/wait."""

    # buffer_load_to_shared produces NaNs on the masked path,
    # so we use global_load_to_shared instead.
    if USE_BUFFER_LOAD:
        _async_load_fn(smemK_buf, k_base, k_offsets)
        _async_load_fn(smemV_buf, v_base, v_offsets)
    else:
        # n-axis masked only on boundary blocks, head-dim only when padded
        # (None => that axis is unmasked).
        offs_kn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, K_LOAD_LAYOUT))
        offs_vn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, V_LOAD_LAYOUT))
        offs_kd = gl.arange(
            0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(1, K_LOAD_LAYOUT)
        )
        offs_vd = gl.arange(
            0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, V_LOAD_LAYOUT)
        )
        if MASK_STEPS:
            k_offs_n = copy_start_n + offs_kn
            v_offs_n = copy_start_n + offs_vn
        else:
            k_offs_n = None
            v_offs_n = None
        if PADDED_HEAD:
            k_offs_d = offs_kd
            v_offs_d = offs_vd
        else:
            k_offs_d = None
            v_offs_d = None
        _async_load_masked_fn(
            smemK_buf, k_base, k_offsets, k_offs_d, k_offs_n, BLOCK_DMODEL, seqlen_k
        )
        _async_load_masked_fn(
            smemV_buf, v_base, v_offsets, v_offs_n, v_offs_d, seqlen_k, BLOCK_DMODEL
        )


@gluon.jit
def _attn_qk_nomask(
    q,
    k,
    qk_scale,
    MFMA_LAYOUT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """QK^T + scale for one already-staged key block (no masking). k is already
    in its MFMA dot-operand layout; returns float32 scores in MFMA_LAYOUT. Split
    out so the caller can pipeline QK^T of block i+1 ahead of the softmax/P@V of
    block i, overlapping the QK MFMA with the softmax."""
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=MFMA_LAYOUT)
    qk = gl.amd.cdna4.mfma(q, k, qk)
    return qk * qk_scale


@gluon.jit
def _attn_qk(
    q,
    k,
    start_n,
    seqlen_q,
    seqlen_k,
    block_max,
    n_extra_tokens,
    offs_m,
    offs_n,
    qk_scale,
    MFMA_LAYOUT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    MASK_STEPS: gl.constexpr,
):
    """QK^T (+ scale + boundary/causal mask) for one already-staged key block.
    Masks are compiled out when MASK_STEPS / IS_CAUSAL are False."""
    qk = _attn_qk_nomask(q, k, qk_scale, MFMA_LAYOUT, BLOCK_M, BLOCK_N)

    if MASK_STEPS or IS_CAUSAL:
        mask = gl.full([BLOCK_M, BLOCK_N], True, dtype=gl.int1, layout=MFMA_LAYOUT)
        if MASK_STEPS:
            # Only the last visible block can be partial (seqlen_k not a multiple
            # of BLOCK_N). mask_partial is selected via bound_cond to stay
            # branch-free.
            bound_cond = (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0)
            size_n = start_n + offs_n
            mask_partial = size_n[None, :] < seqlen_k
            mask = gl.where(bound_cond, mask_partial, mask)
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n + (seqlen_q - seqlen_k)
            causal_mask = offs_m[:, None] >= causal_boundary[None, :]
            mask = mask & causal_mask
        qk = gl.where(mask, qk, float("-inf"))

    return qk


@gluon.jit
def _attn_softmax_pv(acc, l_i, m_i, qk, v, dotP: gl.constexpr):
    """Online-softmax rescale + P@V accumulation for one key block. qk holds the
    (masked) scores, v the value tile in dot-operand layout. Second half of one
    online-softmax step; returns updated (acc, l_i, m_i)."""
    # Fully-masked rows leave m_ij == -inf and produce NaNs here; those rows are
    # zeroed out in the epilogue.
    m_ij = gl.maximum(m_i, gl.max(qk, 1))
    p = gl.exp2(qk - m_ij[:, None])
    alpha = gl.exp2(m_i - m_ij)
    l_ij = gl.sum(p, 1)

    acc = acc * alpha[:, None]

    # TODO: Layout conversion is not trivial
    p = gl.convert_layout(p.to(v.dtype), layout=dotP)
    acc = gl.amd.cdna4.mfma(p, v, acc)

    l_i = l_i * alpha + l_ij
    m_i = m_ij

    return acc, l_i, m_i


@gluon.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_base,
    k_offsets,
    v_base,
    v_offsets,
    smemK,
    smemV,
    stride_kn,
    stride_vn,
    seqlen_k,
    block_min,
    block_max,
    qk_scale,
    MFMA_LAYOUT: gl.constexpr,
    K_LOAD_LAYOUT: gl.constexpr,
    V_LOAD_LAYOUT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    NUM_KV_BUFFERS: gl.constexpr,
):
    """QK-ahead software-pipelined online-softmax loop over full (unmasked) blocks.

    4-stage pipeline running four blocks' work concurrently. Iteration i issues::

        copy(i+3)  ->  readK(i+2)  ->  QK(i+1)  ->  readV(i) + softmax+P@V(i)
        (HBM->LDS)     (LDS->reg)      (MFMA)      (LDS->reg)  (VALU + MFMA)

    Block i+3's async copy is in flight; block i+2's keys are read a full iteration
    before their QK^T to hide ds_read latency; QK^T(i+1) is issued from the carried
    k_nxt ahead of softmax(i) so the MFMA overlaps the softmax VALU.
    Only qk_cur (block i) and k_nxt (block i+1) are loop carried.
    """
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2

    # buffer_load_to_shared produces NaNs on the masked path,
    # so we use global_load_to_shared instead.
    USE_BUFFER_LOAD: gl.constexpr = not PADDED_HEAD

    dotK: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=MFMA_LAYOUT, k_width=8
    )
    dotP: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=MFMA_LAYOUT, k_width=8
    )
    dotV: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=MFMA_LAYOUT, k_width=8
    )

    n_iter = (block_max - block_min) // BLOCK_N

    # Prologue: stage the first three blocks into LDS, read keys of blocks 0/1
    # and compute QK^T(0). Enter the loop with qk_cur = QK(0), k_nxt = K(1) and
    # block 2's copy in flight.
    _issue_kv_copy(
        smemK.index(0),
        smemV.index(0),
        k_base,
        v_base,
        k_offsets,
        v_offsets,
        block_min,
        seqlen_k,
        K_LOAD_LAYOUT,
        V_LOAD_LAYOUT,
        BLOCK_N,
        BLOCK_DMODEL,
        BLOCK_DMODEL_POW2,
        USE_BUFFER_LOAD,
        False,
        PADDED_HEAD,
    )
    gl.amd.cdna4.async_copy.commit_group()
    k_base += BLOCK_N * stride_kn
    v_base += BLOCK_N * stride_vn

    if n_iter > 1:
        _issue_kv_copy(
            smemK.index(1),
            smemV.index(1),
            k_base,
            v_base,
            k_offsets,
            v_offsets,
            block_min + BLOCK_N,
            seqlen_k,
            K_LOAD_LAYOUT,
            V_LOAD_LAYOUT,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            USE_BUFFER_LOAD,
            False,
            PADDED_HEAD,
        )
        gl.amd.cdna4.async_copy.commit_group()
        k_base += BLOCK_N * stride_kn
        v_base += BLOCK_N * stride_vn

    if n_iter > 2:
        _issue_kv_copy(
            smemK.index(2),
            smemV.index(2),
            k_base,
            v_base,
            k_offsets,
            v_offsets,
            block_min + 2 * BLOCK_N,
            seqlen_k,
            K_LOAD_LAYOUT,
            V_LOAD_LAYOUT,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            USE_BUFFER_LOAD,
            False,
            PADDED_HEAD,
        )
        gl.amd.cdna4.async_copy.commit_group()
        k_base += BLOCK_N * stride_kn
        v_base += BLOCK_N * stride_vn
        # blocks 0,1 ready; block 2's copy stays in flight (read at iteration 0).
        gl.amd.cdna4.async_copy.wait_group(1)
    else:
        # <=2 blocks: no block-2 copy, drain everything.
        gl.amd.cdna4.async_copy.wait_group(0)

    k = smemK.index(0).load(dotK)
    qk_cur = _attn_qk_nomask(
        q,
        k,
        qk_scale,
        MFMA_LAYOUT=MFMA_LAYOUT,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    # k_nxt holds block 1's keys for the first in-loop QK^T.
    if n_iter > 1:
        k_nxt = smemK.index(1).load(dotK)
    else:
        k_nxt = k

    for i in range(0, n_iter):
        # V(i): read early so the ds_read overlaps the QK MFMA + softmax VALU
        # before P@V consumes it.
        v_cur = smemV.index(i % NUM_KV_BUFFERS).load(dotV)

        if i + 1 < n_iter:
            # (a) global prefetch: async-copy block i+3. Its buffer last held
            # block i-1, so there is a one-iteration WAR margin.
            if i + 3 < n_iter:
                g_idx = (i + 3) % NUM_KV_BUFFERS
                _issue_kv_copy(
                    smemK.index(g_idx),
                    smemV.index(g_idx),
                    k_base,
                    v_base,
                    k_offsets,
                    v_offsets,
                    block_min + (i + 3) * BLOCK_N,
                    seqlen_k,
                    K_LOAD_LAYOUT,
                    V_LOAD_LAYOUT,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    BLOCK_DMODEL_POW2,
                    USE_BUFFER_LOAD,
                    False,
                    PADDED_HEAD,
                )
                gl.amd.cdna4.async_copy.commit_group()
                k_base += BLOCK_N * stride_kn
                v_base += BLOCK_N * stride_vn

            # (b) wait for block i+2's copy to land before reading its keys.
            if i + 3 < n_iter:
                gl.amd.cdna4.async_copy.wait_group(1)
            elif i + 2 < n_iter:
                gl.amd.cdna4.async_copy.wait_group(0)

            # (c) local K prefetch: read block i+2's keys a full iteration ahead
            # of its QK^T so the ds_read is hidden.
            if i + 2 < n_iter:
                k_rd = smemK.index((i + 2) % NUM_KV_BUFFERS).load(dotK)
            else:
                # No block i+2: keep k_rd defined (unused past this iteration).
                k_rd = k_nxt

            # (d) QK^T of block i+1 from the resident k_nxt, ahead of softmax(i).
            qk_nxt = _attn_qk_nomask(
                q,
                k_nxt,
                qk_scale,
                MFMA_LAYOUT=MFMA_LAYOUT,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

            # (e) softmax + P@V for block i, overlapping the QK MFMA just issued.
            # Carry qk / keys forward one block.
            acc, l_i, m_i = _attn_softmax_pv(acc, l_i, m_i, qk_cur, v_cur, dotP)
            qk_cur = qk_nxt
            k_nxt = k_rd
        else:
            # Final block: pipeline drained, just the softmax + P@V remains.
            acc, l_i, m_i = _attn_softmax_pv(acc, l_i, m_i, qk_cur, v_cur, dotP)

    return acc, l_i, m_i


@gluon.jit
def _attn_fwd_inner_masked(
    acc,
    l_i,
    m_i,
    q,
    k_base,
    k_offsets,
    v_base,
    v_offsets,
    smemK,
    smemV,
    stride_kn,
    stride_vn,
    seqlen_q,
    seqlen_k,
    block_min,
    block_max,
    n_extra_tokens,
    offs_m,
    qk_scale,
    MFMA_LAYOUT: gl.constexpr,
    K_LOAD_LAYOUT: gl.constexpr,
    V_LOAD_LAYOUT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    MASK_STEPS: gl.constexpr,
):
    """Non-pipelined online-softmax loop over the boundary / causal masked blocks.
    The masked tail is only a block or two, so the pipeline overhead is not worth it.
    """
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2
    USE_BUFFER_LOAD: gl.constexpr = (not MASK_STEPS) and (not PADDED_HEAD)

    dotK: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=MFMA_LAYOUT, k_width=8
    )
    dotP: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=MFMA_LAYOUT, k_width=8
    )
    dotV: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=MFMA_LAYOUT, k_width=8
    )

    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, MFMA_LAYOUT))

    n_iter = (block_max - block_min) // BLOCK_N

    for i in range(0, n_iter):
        start_n = block_min + i * BLOCK_N

        # Stage this block into buffer 0 and wait before reading.
        _issue_kv_copy(
            smemK.index(0),
            smemV.index(0),
            k_base,
            v_base,
            k_offsets,
            v_offsets,
            start_n,
            seqlen_k,
            K_LOAD_LAYOUT,
            V_LOAD_LAYOUT,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            USE_BUFFER_LOAD,
            MASK_STEPS,
            PADDED_HEAD,
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)
        k_base += BLOCK_N * stride_kn
        v_base += BLOCK_N * stride_vn

        k = smemK.index(0).load(dotK)
        v = smemV.index(0).load(dotV)

        qk = _attn_qk(
            q,
            k,
            start_n,
            seqlen_q,
            seqlen_k,
            block_max,
            n_extra_tokens,
            offs_m,
            offs_n,
            qk_scale,
            MFMA_LAYOUT=MFMA_LAYOUT,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=IS_CAUSAL,
            MASK_STEPS=MASK_STEPS,
        )
        acc, l_i, m_i = _attn_softmax_pv(acc, l_i, m_i, qk, v, dotP)

    return acc, l_i, m_i


_attn_fwd_repr = make_kernel_repr(
    "_attn_fwd",
    [
        "IS_CAUSAL",
        "NUM_Q_HEADS",
        "NUM_K_HEADS",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_DMODEL",
        "VARLEN",
        "NUM_XCD",
    ],
)


@gluon.jit(repr=_attn_fwd_repr)
def _attn_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    SEQLEN_Q,
    SEQLEN_K,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    NUM_Q_HEADS: gl.constexpr,
    NUM_K_HEADS: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    VARLEN: gl.constexpr,
    BATCH,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    PRELOAD_V: gl.constexpr,
    NUM_XCD: gl.constexpr,
    HEAD_STRIDE_ALIGNED_8: gl.constexpr = False,
    num_warps: gl.constexpr = 4,
):
    RCP_LN2: gl.constexpr = 1.4426950408889634
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2

    # program -> (batch, q_head, query block). SEQLEN_Q is the max query length,
    # so NUM_BLOCKS_M matches the launch grid in both fixed and varlen mode.
    NUM_BLOCKS_M = gl.cdiv(SEQLEN_Q, BLOCK_M)
    pid = gl.program_id(axis=0)
    off_q_head = pid % NUM_Q_HEADS
    # Remap the q-head index across XCDs for better cache locality.
    off_q_head = remap_xcd(off_q_head, NUM_Q_HEADS, NUM_XCD)
    start_m = (pid // NUM_Q_HEADS) % NUM_BLOCKS_M
    off_z = pid // (NUM_Q_HEADS * NUM_BLOCKS_M) % BATCH

    # In varlen mode the lengths come from cu_seqlens and the batch axis is
    # collapsed (stride_*z == 0); in fixed mode use the SEQLEN_Q/SEQLEN_K args.
    if VARLEN:
        cu_seqlens_q_start = gl.load(cu_seqlens_q + off_z)
        seqlen_q = gl.load(cu_seqlens_q + off_z + 1) - cu_seqlens_q_start
        # This query block is entirely past the end of this batch's sequence.
        if start_m * BLOCK_M >= seqlen_q:
            return
        cu_seqlens_k_start = gl.load(cu_seqlens_k + off_z)
        seqlen_k = gl.load(cu_seqlens_k + off_z + 1) - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = SEQLEN_Q
        seqlen_k = SEQLEN_K

    grp_sz: gl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    off_k_head = off_q_head // grp_sz

    # One MFMA layout reused for both matmuls (QK^T and P@V).
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )
    dotQ: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )

    # Coalesced global-load layouts, converted to the dot-operand layouts after
    # loading. Sized to the padded head dim; the real head dim only drives the
    # head-padding masks.
    qLoadLayout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [512 // BLOCK_DMODEL_POW2, BLOCK_DMODEL_POW2 // 8],
        [num_warps, 1],
        [1, 0],
    )
    # K is read transposed as [BLOCK_DMODEL_POW2, BLOCK_N] (head dim contiguous).
    kLoadLayout: gl.constexpr = gl.BlockedLayout(
        [8, 1],
        [BLOCK_DMODEL_POW2 // 8, 512 // BLOCK_DMODEL_POW2],
        [1, num_warps],
        [0, 1],
    )
    vLoadLayout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [512 // BLOCK_DMODEL_POW2, BLOCK_DMODEL_POW2 // 8],
        [num_warps, 1],
        [1, 0],
    )

    # Swizzled shared layouts for the async global->LDS staging of K and V.
    _KV_SHARED: gl.constexpr = _make_kv_shared_layouts(
        BLOCK_DMODEL_POW2, k_ptr.dtype.element_ty.primitive_bitwidth // 8
    )
    kSharedLayout: gl.constexpr = _KV_SHARED[0]
    vSharedLayout: gl.constexpr = _KV_SHARED[1]

    # When the caller guarantees Q/K/V head strides are multiples of 8 elements,
    # the head-axis offset is 16-byte aligned; hinting the multiple lets AxisInfo
    # widen the global loads.
    qh_off = off_q_head * stride_qh
    kh_off = off_k_head * stride_kh
    vh_off = off_k_head * stride_vh
    if HEAD_STRIDE_ALIGNED_8:
        qh_off = gl.multiple_of(qh_off, 8)
        kh_off = gl.multiple_of(kh_off, 8)
        vh_off = gl.multiple_of(vh_off, 8)

    # Load Q (stays resident for the whole key loop).
    offs_qm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, qLoadLayout))
    offs_qd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, qLoadLayout))
    q_base = (
        q_ptr
        + off_z * stride_qz
        + qh_off
        + cu_seqlens_q_start * stride_qm
        + start_m * BLOCK_M * stride_qm
    )
    q_offsets = (offs_qm[:, None] * stride_qm + offs_qd[None, :] * stride_qk).to(
        gl.int32
    )
    q_mask = (start_m * BLOCK_M + offs_qm)[:, None] < seqlen_q
    if PADDED_HEAD:
        q_mask = q_mask & (offs_qd[None, :] < BLOCK_DMODEL)
    # Cache Q at .cg when a single Q block spans at least one full head.
    if BLOCK_M >= NUM_Q_HEADS:
        q_cache_mod: gl.constexpr = ".cg"
    else:
        q_cache_mod: gl.constexpr = ""
    q = gl.amd.cdna4.buffer_load(
        ptr=q_base, offsets=q_offsets, mask=q_mask, other=0.0, cache=q_cache_mod
    )
    q = gl.convert_layout(q, layout=dotQ)

    offs_kd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(1, kLoadLayout))
    offs_kn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kLoadLayout))
    k_base = k_ptr + off_z * stride_kz + kh_off + cu_seqlens_k_start * stride_kn
    k_offsets = (offs_kd[:, None] * stride_kk + offs_kn[None, :] * stride_kn).to(
        gl.int32
    )

    offs_vn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, vLoadLayout))
    offs_vd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, vLoadLayout))
    v_base = v_ptr + off_z * stride_vz + vh_off + cu_seqlens_k_start * stride_vn
    v_offsets = (offs_vn[:, None] * stride_vn + offs_vd[None, :] * stride_vk).to(
        gl.int32
    )

    # Shared-memory tiles for the async K/V staging. Quad-buffered for
    # _attn_fwd_inner's 4-stage pipeline (four blocks live per iteration).
    NUM_KV_BUFFERS: gl.constexpr = 4
    smemK = gl.allocate_shared_memory(
        k_ptr.dtype.element_ty,
        [NUM_KV_BUFFERS, BLOCK_DMODEL_POW2, BLOCK_N],
        kSharedLayout,
    )
    smemV = gl.allocate_shared_memory(
        v_ptr.dtype.element_ty,
        [NUM_KV_BUFFERS, BLOCK_N, BLOCK_DMODEL_POW2],
        vSharedLayout,
    )

    # online-softmax state
    m_i = gl.full(
        [BLOCK_M], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, mfmaLayout)
    )
    l_i = gl.full(
        [BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, mfmaLayout)
    )
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=gl.float32, layout=mfmaLayout)

    qk_scale = sm_scale * RCP_LN2

    # Query positions used for the causal mask, in the MFMA result layout.
    offs_m = start_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout)
    )

    # Classify key blocks: full (mask-free) vs masked (boundary/causal).
    n_blocks = gl.cdiv(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_causal = gl.cdiv(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )
        n_blocks = min(n_blocks, n_blocks_causal)

        if n_blocks <= 0:
            offs_od = gl.arange(
                0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, mfmaLayout)
            )
            offs_rm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout))
            o_base = (
                o_ptr
                + off_z * stride_oz
                + off_q_head * stride_oh
                + cu_seqlens_q_start * stride_om
                + start_m * BLOCK_M * stride_om
            )
            o_offsets = (
                offs_rm[:, None] * stride_om + offs_od[None, :] * stride_on
            ).to(gl.int32)
            zeros = gl.zeros(
                [BLOCK_M, BLOCK_DMODEL_POW2],
                dtype=o_ptr.dtype.element_ty,
                layout=mfmaLayout,
            )
            o_mask = offs_m[:, None] < seqlen_q
            if PADDED_HEAD:
                o_mask = o_mask & (offs_od[None, :] < BLOCK_DMODEL)
            gl.amd.cdna4.buffer_store(zeros, ptr=o_base, offsets=o_offsets, mask=o_mask)
            return

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = (not padded_block_k) and (seqlen_q % BLOCK_M == 0)

    if IS_CAUSAL:
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        masked_blocks = padded_block_k

    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N

    # Full blocks: no boundary mask, no causal mask.
    if n_full_blocks > 0:
        block_max = n_full_blocks * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_base,
            k_offsets,
            v_base,
            v_offsets,
            smemK,
            smemV,
            stride_kn,
            stride_vn,
            seqlen_k,
            block_min,
            block_max,
            qk_scale,
            MFMA_LAYOUT=mfmaLayout,
            K_LOAD_LAYOUT=kLoadLayout,
            V_LOAD_LAYOUT=vLoadLayout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
            NUM_KV_BUFFERS=NUM_KV_BUFFERS,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    # Remaining blocks carry the boundary / causal masking (non-pipelined path).
    if masked_blocks > 0:
        k_base += n_full_blocks * BLOCK_N * stride_kn
        v_base += n_full_blocks * BLOCK_N * stride_vn
        acc, l_i, m_i = _attn_fwd_inner_masked(
            acc,
            l_i,
            m_i,
            q,
            k_base,
            k_offsets,
            v_base,
            v_offsets,
            smemK,
            smemV,
            stride_kn,
            stride_vn,
            seqlen_q,
            seqlen_k,
            block_min,
            block_max,
            n_extra_tokens,
            offs_m,
            qk_scale,
            MFMA_LAYOUT=mfmaLayout,
            K_LOAD_LAYOUT=kLoadLayout,
            V_LOAD_LAYOUT=vLoadLayout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
            IS_CAUSAL=IS_CAUSAL,
            MASK_STEPS=True,
        )

    # epilogue: normalize and write
    acc = acc / l_i[:, None]

    offs_rm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout))
    offs_om = start_m * BLOCK_M + offs_rm
    offs_od = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, mfmaLayout))

    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    if IS_CAUSAL:
        if (causal_start_idx > start_m_idx) and (causal_start_idx < end_m_idx):
            out_mask_boundary = gl.full(
                [BLOCK_DMODEL_POW2],
                causal_start_idx,
                dtype=gl.int32,
                layout=gl.SliceLayout(0, mfmaLayout),
            )
            mask_m_offsets = start_m_idx + gl.arange(
                0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout)
            )
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            acc = gl.where(out_ptrs_mask, acc, 0.0)

    out = acc.to(o_ptr.dtype.element_ty)

    o_base = (
        o_ptr
        + off_z * stride_oz
        + off_q_head * stride_oh
        + cu_seqlens_q_start * stride_om
        + start_m * BLOCK_M * stride_om
    )
    o_offsets = (offs_rm[:, None] * stride_om + offs_od[None, :] * stride_on).to(
        gl.int32
    )

    overflow_size = end_m_idx - seqlen_q
    out_mask = gl.full([BLOCK_M, 1], True, dtype=gl.int1, layout=mfmaLayout)
    if overflow_size > 0:
        out_mask = out_mask & (offs_om[:, None] < seqlen_q)
    if PADDED_HEAD:
        out_mask = out_mask & (offs_od[None, :] < BLOCK_DMODEL)
    gl.amd.cdna4.buffer_store(out, ptr=o_base, offsets=o_offsets, mask=out_mask)


def _validate_and_launch(
    q,
    k,
    v,
    o,
    sm_scale,
    causal,
    seqlen_q,
    seqlen_k,
    num_q_heads,
    num_k_heads,
    head_dim,
    batch,
    q_strides,
    k_strides,
    v_strides,
    o_strides,
    cu_seqlens_q,
    cu_seqlens_k,
    varlen,
    BLOCK_M,
    BLOCK_N,
):
    """Shared validation + launch for the fixed-length and varlen wrappers.
    *_strides are 4-tuples in (batch, head, seq, head_dim) order; seqlen_q is the
    max query length used to size the grid."""
    assert is_gluon_available(), (
        f"Gluon MHA backend requires one of {_GLUON_SUPPORTED_ARCHS} with "
        f"Triton>=3.6 (arch={get_arch()!r}, triton={triton.__version__})"
    )
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "head_dim mismatch"
    assert (
        num_q_heads % num_k_heads == 0
    ), "num_q_heads must be divisible by num_k_heads"
    # Pad head dim up to a power of 2 (>=32): MFMA 16x16x32 needs the contraction
    # to be a multiple of 32 and the blocked load layouts need it to divide 512.
    BLOCK_DMODEL_POW2 = max(triton.next_power_of_2(head_dim), 32)
    assert BLOCK_N % 32 == 0, "BLOCK_N must be a multiple of 32"

    head_stride_aligned_8 = (
        q_strides[1] % 8 == 0 and k_strides[1] % 8 == 0 and v_strides[1] % 8 == 0
    )

    grid = (batch * num_q_heads * triton.cdiv(seqlen_q, BLOCK_M), 1)

    _attn_fwd[grid](
        q,
        k,
        v,
        o,
        sm_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen_q,
        seqlen_k,
        *q_strides,  #
        *k_strides,  #
        *v_strides,  #
        *o_strides,  #
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        IS_CAUSAL=causal,
        VARLEN=varlen,
        BATCH=batch,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=head_dim,
        BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
        PRELOAD_V=True,
        NUM_XCD=get_num_xcds(),
        HEAD_STRIDE_ALIGNED_8=head_stride_aligned_8,
        num_warps=4,
        waves_per_eu=2,
    )
    return o


def flash_attn_fwd(
    q,
    k,
    v,
    causal=False,
    sm_scale=None,
    o=None,
    BLOCK_M=128,
    BLOCK_N=64,
):
    """Gluon flash-attention forward (fixed-length / padded batch).

    Arguments:
        q: (batch, seqlen_q, num_q_heads, head_dim)
        k: (batch, seqlen_k, num_k_heads, head_dim)
        v: (batch, seqlen_k, num_k_heads, head_dim)
        causal: whether to apply a (bottom-right aligned) causal mask.
        sm_scale: QK^T scale. Defaults to 1 / sqrt(head_dim).
    Return:
        o: (batch, seqlen_q, num_q_heads, head_dim)
    """
    batch, seqlen_q, num_q_heads, head_dim = q.shape
    _, seqlen_k, num_k_heads, _ = k.shape

    if sm_scale is None:
        sm_scale = head_dim ** (-0.5)
    if o is None:
        o = torch.empty_like(q)

    # (batch, head, seq, head_dim) stride order expected by the kernel.
    q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
    k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
    v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
    o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

    return _validate_and_launch(
        q,
        k,
        v,
        o,
        sm_scale,
        causal,
        seqlen_q,
        seqlen_k,
        num_q_heads,
        num_k_heads,
        head_dim,
        batch,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        varlen=False,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


def flash_attn_varlen_fwd(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal=False,
    sm_scale=None,
    o=None,
    BLOCK_M=128,
    BLOCK_N=64,
):
    """Gluon flash-attention forward for ragged (varlen) batches.

    Q/K/V are packed token sequences without a batch dimension; the per-sequence
    boundaries are given by the cumulative-length tensors.

    Arguments:
        q: (total_q, num_q_heads, head_dim), all query tokens across the batch.
        k: (total_k, num_k_heads, head_dim), all key tokens across the batch.
        v: (total_k, num_k_heads, head_dim), all value tokens across the batch.
        cu_seqlens_q: (batch + 1,) int32 cumulative query sequence lengths.
        cu_seqlens_k: (batch + 1,) int32 cumulative key sequence lengths.
        max_seqlen_q: maximum query sequence length in the batch.
        max_seqlen_k: maximum key sequence length in the batch.
        causal: whether to apply a (bottom-right aligned) causal mask.
        sm_scale: QK^T scale. Defaults to 1 / sqrt(head_dim).
    Return:
        o: (total_q, num_q_heads, head_dim)
    """
    _, num_q_heads, head_dim = q.shape
    _, num_k_heads, _ = k.shape
    batch = cu_seqlens_q.numel() - 1
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)

    if sm_scale is None:
        sm_scale = head_dim ** (-0.5)
    if o is None:
        o = torch.empty_like(q)

    # Packed [total_tokens, head, head_dim] layout: the batch axis is collapsed,
    # so its stride is 0 and the per-sequence row offset comes from cu_seqlens.
    q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
    k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
    v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
    o_strides = (0, o.stride(1), o.stride(0), o.stride(2))

    return _validate_and_launch(
        q,
        k,
        v,
        o,
        sm_scale,
        causal,
        max_seqlen_q,
        max_seqlen_k,
        num_q_heads,
        num_k_heads,
        head_dim,
        batch,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        varlen=True,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
