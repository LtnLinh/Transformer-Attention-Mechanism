import math
from functools import lru_cache

import numpy as np
import torch
from numba import cuda, float32

from gpu_base import GpuPipeline

TILE = 16      # tile edge: 16 query rows per block, 16-wide staging chunks
TILE_P = 17    # padded row width for staged tiles (dodges bank conflicts)
BK = 64        # keys swept per iteration: four per lane, the register micro-tile

# finite stand-in for -inf on masked keys: exp() of it is 0 and it dodges the
# (-inf) - (-inf) = nan trap when a tile holds no real keys
_NEG_BIG = float32(-3.0e38)
_M_INIT = float32(-math.inf)


@lru_cache(maxsize=None)
def _flash_kernel(D):
    """Build the FlashAttention-1 forward kernel for a fixed model width D —
    numba wants compile-time-constant shared/local array shapes.

    Each lane owns FOUR keys of the 64-key sweep (columns tx, tx+16, tx+32,
    tx+48) — a register micro-tile. That ratio is what the T4 counters ask
    for: without it the scalar pipeline is bound by its own instruction
    stream (~2 shared loads per FMA from 2 warps per scheduler — not by
    occupancy, traffic, or barriers), and one staged q element feeding four
    FMAs is what moves it. A 64-key sweep also runs ONE online-softmax
    merge and a quarter of the barriers where four 16-key tiles ran four."""
    assert D % TILE == 0, "flash kernel assumes d_model is a multiple of 16"
    CH = D // TILE   # output columns owned by each thread (cols tx, tx+16, ...)
    D_P = D + 1      # padded Q row: plain name, same shared.array constraint
    BK_P = BK + 1
    KW = BK // TILE  # keys per lane

    @cuda.jit
    def flash(q, k, v, scale, out):
        tx = cuda.threadIdx.x          # lane: key column within each 16-chunk
        ty = cuda.threadIdx.y          # query row within the tile
        base = cuda.blockIdx.x * TILE  # first query row of this block
        N = q.shape[0]

        qs = cuda.shared.array((TILE, D_P), float32)   # resident Q tile
        ps = cuda.shared.array((TILE, BK_P), float32)  # P~ = exp(S - m~), 64 keys
        ks = cuda.shared.array((BK, TILE_P), float32)  # K chunk [key, dim]
        vs = cuda.shared.array((BK, TILE_P), float32)  # V chunk [key, col]

        # Stage the block's Q tile once (each thread strides one row); it
        # stays on-chip for the whole K/V sweep — the FlashAttention move
        # that makes Q traffic O(N D) instead of O(N^2 D).
        for c in range(tx, D, TILE):
            qs[ty, c] = q[base + ty, c] if base + ty < N else float32(0.)

        acc = cuda.local.array(CH, float32)  # O[base+ty, tx::TILE] — kept
        for c in range(CH):                  # normalised after every sweep
            acc[c] = float32(0.)
        m = _M_INIT      # running row max
        l = float32(0.)  # running row sum of exponentials

        for jt in range(0, N, BK):
            # --- scores for this lane's 4 keys: one qs load feeds 4 FMAs
            s0 = float32(0.)
            s1 = float32(0.)
            s2 = float32(0.)
            s3 = float32(0.)
            for i in range(CH):
                dt = i * TILE
                for w in range(KW):  # stage 64 keys x 16 dims, 4 rows/lane
                    kr = jt + ty + w * TILE
                    ks[ty + w * TILE, tx] = k[kr, dt + tx] if kr < N else float32(0.)
                cuda.syncthreads()
                for j in range(TILE):
                    a = qs[ty, dt + j]
                    s0 += a * ks[tx, j]
                    s1 += a * ks[tx + TILE, j]
                    s2 += a * ks[tx + 2 * TILE, j]
                    s3 += a * ks[tx + 3 * TILE, j]
                cuda.syncthreads()
            s0 *= scale
            s1 *= scale
            s2 *= scale
            s3 *= scale
            if jt + tx >= N:  # masked key: exp() contributes exactly 0
                s0 = _NEG_BIG
            if jt + tx + TILE >= N:
                s1 = _NEG_BIG
            if jt + tx + 2 * TILE >= N:
                s2 = _NEG_BIG
            if jt + tx + 3 * TILE >= N:
                s3 = _NEG_BIG

            # --- sweep row max and sum: local reduce over the lane's 4 keys,
            # then a 16-lane shuffle butterfly (a warp holds rows ty, ty+1;
            # xor offsets 8/4/2/1 stay inside each row's 16-lane half)
            mt = s0
            if s1 > mt:
                mt = s1
            if s2 > mt:
                mt = s2
            if s3 > mt:
                mt = s3
            o = cuda.shfl_xor_sync(0xFFFFFFFF, mt, 8)
            if o > mt:
                mt = o
            o = cuda.shfl_xor_sync(0xFFFFFFFF, mt, 4)
            if o > mt:
                mt = o
            o = cuda.shfl_xor_sync(0xFFFFFFFF, mt, 2)
            if o > mt:
                mt = o
            o = cuda.shfl_xor_sync(0xFFFFFFFF, mt, 1)
            if o > mt:
                mt = o
            p0 = math.exp(s0 - mt)  # P~ uses the *sweep* max (FA-1's Algorithm 1)
            p1 = math.exp(s1 - mt)
            p2 = math.exp(s2 - mt)
            p3 = math.exp(s3 - mt)
            lt = (p0 + p1) + (p2 + p3)
            lt += cuda.shfl_xor_sync(0xFFFFFFFF, lt, 8)
            lt += cuda.shfl_xor_sync(0xFFFFFFFF, lt, 4)
            lt += cuda.shfl_xor_sync(0xFFFFFFFF, lt, 2)
            lt += cuda.shfl_xor_sync(0xFFFFFFFF, lt, 1)
            ps[ty, tx] = p0  # published block-wide by the first PV barrier
            ps[ty, tx + TILE] = p1
            ps[ty, tx + 2 * TILE] = p2
            ps[ty, tx + 3 * TILE] = p3

            # FA-1 online-softmax merge: fold the sweep's (m~, l~) into the
            # running (m, l) and rescale AND renormalise the output block
            # this very sweep — no deferral; acc always holds the softmax of
            # everything seen so far. First sweep: exp(-inf - m_new) = 0.
            mn = mt if mt > m else m
            alpha = math.exp(m - mn)
            beta = math.exp(mt - mn)
            ln = l * alpha + lt * beta
            scale_old = l * alpha / ln  # rescales the already-normalised acc
            scale_new = beta / ln       # normalises this sweep's P~ V

            # --- acc = scale_old * acc + scale_new * (P~ @ V), 64 keys per
            # staged 16-column V chunk, 4 independent partial sums
            for c in range(CH):
                for w in range(KW):
                    kr = jt + ty + w * TILE
                    vs[ty + w * TILE, tx] = v[kr, c * TILE + tx] if kr < N else float32(0.)
                cuda.syncthreads()
                q0 = float32(0.)
                q1 = float32(0.)
                q2 = float32(0.)
                q3 = float32(0.)
                for j in range(0, BK, 4):
                    q0 += ps[ty, j] * vs[j, tx]
                    q1 += ps[ty, j + 1] * vs[j + 1, tx]
                    q2 += ps[ty, j + 2] * vs[j + 2, tx]
                    q3 += ps[ty, j + 3] * vs[j + 3, tx]
                acc[c] = acc[c] * scale_old + ((q0 + q1) + (q2 + q3)) * scale_new
                cuda.syncthreads()
            m = mn
            l = ln

        # acc is already the normalised softmax-weighted sum — write as-is
        if base + ty < N:
            for c in range(CH):
                out[base + ty, c * TILE + tx] = acc[c]

    return flash


class GpuV3(GpuPipeline):
    """V3 — FlashAttention-1 forward: tiled + online softmax + one fused
    kernel, and the N x N score matrix never exists in any memory.

    One thread block per 16 query rows: the Q tile is staged in shared
    memory once and stays resident while the block sweeps K/V 64 keys at a
    time. Each lane owns four of those keys — a register micro-tile, the
    one scheduling idea the T4 measurements demanded: it cuts the shared
    loads per FMA (the scalar pipeline's real limit) and runs one
    Algorithm-1 merge per sweep where 16-key tiles ran four. The sweep's
    row (max, sum) reduce over warp shuffles, the running statistics merge,
    and the output block is rescaled and renormalised immediately, so it
    always holds the exact softmax-weighted sum of every key seen so far.
    Extra footprint is one [N, D] output — O(N) — and Q/K/V are each read
    once per Q-block sweep."""

    def _attend(self, q, k, v):
        N, D = q.shape
        out = torch.empty((N, D), device=q.device)
        kernel = _flash_kernel(D)
        # np.float32 keeps the kernel arithmetic in fp32: a python float
        # scale is typed float64 and would silently promote the hot loops
        kernel[math.ceil(N / TILE), (TILE, TILE)](
            q, k, v, np.float32(1.0 / D ** 0.5), out)
        return out
