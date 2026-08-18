import math
from functools import lru_cache

import numpy as np
import torch
from numba import cuda, float32

from gpu_base import GpuPipeline

# The paper's M (Algorithm 1, line 1): on-chip SRAM size. The T4 has 64 KB
# of shared memory per SM = 16384 fp32 elements.
M_FLOATS = 16384
W = 32  # column workers per block row: one warp strides the d axis

# finite stand-in for -inf on masked keys: exp() of it is 0 and it dodges the
# (-inf) - (-inf) = nan trap
_NEG_BIG = float32(-3.0e38)


@lru_cache(maxsize=None)
def _fa1_kernel(D):
    """Build Algorithm 1's inner body (lines 6-13) for one (i, j) block pair.

    This kernel is deliberately a line-by-line transcription of the paper —
    explainability over speed. One thread computes one score element; the
    row max and sum are plain shared-memory scans; there is no register
    micro-tiling, no warp shuffling, no prefetching. The known costs of
    that choice on a T4 (idle lanes in the score phase, bank conflicts on
    the K reads, per-launch dispatch for the outer loop) are accepted.
    """
    # Line 1: Bc = ceil(M / 4d), Br = min(ceil(M / 4d), d). The budget means
    # exactly this: four Bc x d fp32 blocks (Kj, Vj, Qi, Oi) fill the SRAM.
    BC = max(1, M_FLOATS // (4 * D))  # 4d divides M here, so floor == ceil
    BR = min(BC, D)
    # Three of the four blocks live in shared memory below (Qi, Kj, Vj —
    # exactly the 48 KB a CUDA block may allocate statically); the fourth,
    # Oi, is "loaded on chip" (line 8) into registers instead, one column
    # slice per thread, and written back at line 12.
    assert 3 * BC * D * 4 <= 49152, "Qi, Kj, Vj must fit in static shared"

    @cuda.jit
    def fa1_inner(q, k, v, o, l, m, j, scale):
        i = cuda.blockIdx.x   # line 7: the inner loop over i — its
        #                       iterations touch disjoint rows of O, l, m,
        #                       so they run as independent thread blocks
        tx = cuda.threadIdx.x          # column worker within the row
        ty = cuda.threadIdx.y          # row within the Br-row block
        row = i * BR + ty              # this row of Q, O, l, m
        kbase = j * BC                 # first key of block j
        N = q.shape[0]

        qi = cuda.shared.array((BR, D), float32)  # line 8: load Qi
        kj = cuda.shared.array((BC, D), float32)  # line 6: load Kj
        vj = cuda.shared.array((BC, D), float32)  # line 6: load Vj

        for c in range(tx, D, W):
            qi[ty, c] = q[row, c] if row < N else float32(0.)
            kj[ty, c] = k[kbase + ty, c] if kbase + ty < N else float32(0.)
            vj[ty, c] = v[kbase + ty, c] if kbase + ty < N else float32(0.)
        cuda.syncthreads()

        # Line 9: S_ij = tau * Qi @ Kj^T — one thread per score element
        # (threads with tx >= Bc sit this phase out).
        s = _NEG_BIG
        if tx < BC:
            s = float32(0.)
            for kk in range(D):
                s += qi[ty, kk] * kj[tx, kk]
            s *= scale
            if kbase + tx >= N:
                s = _NEG_BIG  # masked key: exp() contributes exactly 0
        cuda.syncthreads()  # everyone is done reading Kj...
        if tx < BC:
            kj[ty, tx] = s  # ...so its SRAM slot is free: recycle the first
        cuda.syncthreads()  # Br x Bc entries to hold S_ij, then P~_ij

        # Line 10: m~ = rowmax(S), P~ = exp(S - m~), l~ = rowsum(P~).
        mt = _NEG_BIG
        for c in range(BC):
            if kj[ty, c] > mt:
                mt = kj[ty, c]
        cuda.syncthreads()  # all scans done before S is overwritten by P~
        if tx < BC:
            kj[ty, tx] = math.exp(s - mt)
        cuda.syncthreads()
        lt = float32(0.)
        for c in range(BC):
            lt += kj[ty, c]

        # Line 8 (rest) + line 11: load m_i, l_i from HBM and merge —
        # m_new = max(m_i, m~), l_new = e^(m_i-m_new) l_i + e^(m~-m_new) l~.
        mi = m[row] if row < N else _NEG_BIG
        li = l[row] if row < N else float32(0.)
        mn = mt if mt > mi else mi
        alpha = math.exp(mi - mn)
        beta = math.exp(mt - mn)
        ln = li * alpha + lt * beta

        if row < N:
            # Line 12: O_i <- diag(l_new)^-1 (diag(l_i) e^(m_i-m_new) O_i
            #                                 + e^(m~-m_new) P~_ij Vj),
            # read-modify-written in HBM, one column slice per thread.
            for c in range(tx, D, W):
                pv = float32(0.)
                for cc in range(BC):
                    pv += kj[ty, cc] * vj[cc, c]
                o[row, c] = (li * alpha * o[row, c] + beta * pv) / ln
            if tx == 0:
                l[row] = ln  # line 13: write l_i, m_i back to HBM
                m[row] = mn

    return fa1_inner, BR, BC


class GpuV3(GpuPipeline):
    """V3 — FlashAttention-1 forward, Algorithm 1 of the paper transcribed
    as literally as the hardware allows, preferring explainability over
    speed. The host runs lines 2 and 5; the kernel is lines 6-13.

    Structure, mapped to the paper: O, l (row exp-sums) and m (row maxes)
    are initialised in HBM (line 2) and updated there after every (i, j)
    block pair — nothing accumulates in registers across the sweep. The
    outer loop over K/V blocks j is a host loop (line 5, one kernel launch
    per j; same-stream launches serialize, preserving the paper's order);
    the inner loop over Q blocks i (line 7) becomes the launch grid, since
    its iterations touch disjoint rows. Block sizes come from line 1's
    SRAM budget: Bc = Br = M/4d = 8 on the T4's 64 KB. The N x N matrix
    never exists; the extra footprint is O(N): the output plus the l and m
    vectors.

    What this costs: with d = 512 and 16K fp32 of SRAM, the paper's own
    IO bound Theta(N^2 d^2 / M) is no longer small — O is re-read and
    re-written once per K/V block — so this V3 spends bandwidth to hold
    its O(N) footprint, and V2 outruns it at large N. The fast variant
    (register micro-tile, one merge per 64-key sweep, ~1.7x) lives in the
    repo history; this version keeps the paper on the page."""

    def _attend(self, q, k, v):
        N, D = q.shape
        kernel, BR, BC = _fa1_kernel(D)
        # Line 2: O = 0, l = 0, m = -inf, all in HBM.
        out = torch.zeros((N, D), device=q.device)
        l = torch.zeros(N, device=q.device)
        m = torch.full((N,), -math.inf, device=q.device)
        Tr = math.ceil(N / BR)
        Tc = math.ceil(N / BC)
        # np.float32 keeps the kernel arithmetic in fp32: a python float
        # scale is typed float64 and would silently promote the hot loops
        tau = np.float32(1.0 / D ** 0.5)
        for j in range(Tc):  # line 5: the outer loop over K/V blocks
            kernel[Tr, (W, BR)](q, k, v, out, l, m, j, tau)
        return out
