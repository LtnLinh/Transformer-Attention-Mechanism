import math
from functools import lru_cache

import numpy as np
import torch
from numba import cuda, float32

from gpu_base import GpuPipeline

TILE = 16      # tile edge: 16 query rows x 16 key columns per block
TILE_P = 17    # padded row width for the score tile (dodges bank conflicts)

# finite stand-in for -inf on masked keys: exp() of it is 0 and it dodges the
# (-inf) - (-inf) = nan trap when a tile holds no real keys
_NEG_BIG = float32(-3.0e38)
_M_INIT = float32(-math.inf)


@lru_cache(maxsize=None)
def _flash_kernel(D):
    """Build the FlashAttention-1 forward kernel for a fixed model width D —
    numba wants compile-time-constant shared/local array shapes."""
    CH = D // TILE  # output columns owned by each thread (cols tx, tx+16, ...)
    D_P = D + 1     # padded Q row: plain name, same shared.array constraint

    @cuda.jit
    def flash(q, k, v, scale, out):
        tx = cuda.threadIdx.x          # key column within the tile
        ty = cuda.threadIdx.y          # query row within the tile
        base = cuda.blockIdx.x * TILE  # first query row of this block
        N = q.shape[0]

        qs = cuda.shared.array((TILE, D_P), float32)    # resident Q tile
        ps = cuda.shared.array((TILE, TILE_P), float32)  # S, then P~ = exp(S - m~)
        ks = cuda.shared.array((TILE, TILE_P), float32)  # K d-chunk (V2's idiom)
        vs = cuda.shared.array((TILE, TILE_P), float32)  # V column-chunk

        # Stage the block's Q tile once (each thread strides one row); it
        # stays on-chip for the whole K/V sweep — the FlashAttention move
        # that makes Q traffic O(N D) instead of O(N^2 D).
        for c in range(tx, D, TILE):
            qs[ty, c] = q[base + ty, c] if base + ty < N else float32(0.)
        cuda.syncthreads()

        acc = cuda.local.array(CH, float32)  # O[base+ty, tx::TILE] — kept
        for c in range(CH):                  # normalised after every tile (FA-1)
            acc[c] = float32(0.)
        m = _M_INIT      # running row max
        l = float32(0.)  # running row sum of exponentials

        for jt in range(0, N, TILE):
            # S[ty, tx] = scale * dot(q[base+ty], k[jt+tx]) — K staged in
            # 16-wide d-chunks in shared memory, exactly V2's tiling idiom
            s = float32(0.)
            for dt in range(0, D, TILE):
                ks[ty, tx] = k[jt + ty, dt + tx] if jt + ty < N else float32(0.)
                cuda.syncthreads()
                for j in range(TILE):
                    s += qs[ty, dt + j] * ks[tx, j]
                cuda.syncthreads()
            s *= scale
            if jt + tx >= N:
                s = _NEG_BIG  # masked key: exp() contributes exactly 0
            ps[ty, tx] = s
            cuda.syncthreads()

            # tile row max: every lane scans its row's 16 scores — redundant
            # but uniform, so no divergence and no reduction machinery
            mt = _NEG_BIG
            for j in range(TILE):
                if ps[ty, j] > mt:
                    mt = ps[ty, j]
            p = math.exp(s - mt)  # P~ uses the *tile* max (FA-1's Algorithm 1)
            cuda.syncthreads()
            ps[ty, tx] = p
            cuda.syncthreads()
            lt = float32(0.)      # tile row sum, same redundant scan
            for j in range(TILE):
                lt += ps[ty, j]

            # FA-1 online-softmax merge: fold the tile's (m~, l~) into the
            # running (m, l) and rescale AND renormalise the output block this
            # very tile — no deferral; acc always holds the softmax of
            # everything seen so far. First tile: exp(-inf - m_new) = 0.
            mn = mt if mt > m else m
            alpha = math.exp(m - mn)
            beta = math.exp(mt - mn)
            ln = l * alpha + lt * beta
            scale_old = l * alpha / ln  # rescales the already-normalised acc
            scale_new = beta / ln       # normalises this tile's P~ V

            # acc = scale_old * acc + scale_new * (P~ @ V_tile); V staged one
            # 16-column chunk at a time, same idiom
            for c in range(CH):
                vs[ty, tx] = v[jt + ty, c * TILE + tx] if jt + ty < N else float32(0.)
                cuda.syncthreads()
                pv = float32(0.)
                for j in range(TILE):
                    pv += ps[ty, j] * vs[j, tx]
                acc[c] = acc[c] * scale_old + pv * scale_new
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

    One thread block per 16 query rows: the Q tile is staged in shared memory
    once and stays resident while the block sweeps the K/V tiles; K and V are
    staged 16-wide chunk by chunk with the same shared-memory idiom V2 uses —
    no prefetching, no unrolling, no micro-tiles. Per 16x16 tile the block
    computes the score tile, takes its row max and sum, and applies
    Algorithm 1's update: the running (max, sum) statistics merge with the
    tile's and the output block is rescaled and renormalised immediately, so
    it always holds the exact softmax-weighted sum of every key seen so far.
    Extra footprint is one [N, D] output — O(N) — and Q/K/V are each read
    once per Q-block sweep.
    """

    def _attend(self, q, k, v):
        N, D = q.shape
        out = torch.empty((N, D), device=q.device)
        kernel = _flash_kernel(D)
        # np.float32 keeps the kernel arithmetic in fp32: a python float
        # scale is typed float64 and would silently promote the hot loops
        kernel[math.ceil(N / TILE), (TILE, TILE)](
            q, k, v, np.float32(1.0 / D ** 0.5), out)
        return out
