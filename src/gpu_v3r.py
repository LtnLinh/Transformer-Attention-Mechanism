import math
from functools import lru_cache

import numpy as np
import torch
from numba import cuda, float32

from gpu_base import GpuPipeline

TILE = 16      # tile edge: 16 query rows x 16 key columns per block
TILE_P = 17    # padded row width (dodges shared-memory bank conflicts)

_NEG_BIG = float32(-3.0e38)
_M_INIT = float32(-math.inf)


@lru_cache(maxsize=None)
def _flash_reg_kernel(D):
    """V3's FA-1 forward restructured against its measured stall profile
    (33% global-load latency, 19% serial-FMA dependency, at 25% occupancy —
    latency-bound, so this attacks latency per warp instead of occupancy):

    - register prefetch + double-buffered staging: the next K/V chunk's
      global load issues before the current chunk's math, so its latency
      overlaps compute instead of a barrier;
    - the 16-FMA reductions split into 4 independent partial accumulators
      (fp32 addition is not reassociable, so the compiler cannot do this);
    - tile softmax stats (row max / sum) via warp shuffles — a warp holds
      two score rows, one per 16-lane half — replacing shared-memory scans;
    - one barrier per staged chunk instead of two: writes alternate buffers,
      and the next chunk's barrier orders buffer reuse two chunks out.

    Same math, same O(N) footprint, same 16x16 block as V3."""
    assert D % TILE == 0, "flash kernel assumes d_model is a multiple of 16"
    CH = D // TILE  # output columns owned by each thread (cols tx, tx+16, ...)
    D_P = D + 1

    @cuda.jit
    def flashr(q, k, v, scale, out):
        tx = cuda.threadIdx.x          # key column within the tile
        ty = cuda.threadIdx.y          # query row within the tile
        base = cuda.blockIdx.x * TILE  # first query row of this block
        N = q.shape[0]

        qs = cuda.shared.array((TILE, D_P), float32)       # resident Q tile
        ps = cuda.shared.array((TILE, TILE_P), float32)    # P~ = exp(S - m~)
        ks = cuda.shared.array((2, TILE, TILE_P), float32)  # K chunk, 2 buffers
        vs = cuda.shared.array((2, TILE, TILE_P), float32)  # V chunk, 2 buffers

        for c in range(tx, D, TILE):
            qs[ty, c] = q[base + ty, c] if base + ty < N else float32(0.)
        # no barrier here: the first staging barrier below also publishes qs

        acc = cuda.local.array(CH, float32)  # O[base+ty, tx::TILE], normalised
        for c in range(CH):
            acc[c] = float32(0.)
        m = _M_INIT      # running row max
        l = float32(0.)  # running row sum of exponentials

        for jt in range(0, N, TILE):
            krow = jt + ty   # the K/V row this thread stages
            valid = krow < N

            # --- S[ty, tx]: prefetch-pipelined, 4 independent partials ---
            rk = k[krow, tx] if valid else float32(0.)
            s0 = float32(0.)
            s1 = float32(0.)
            s2 = float32(0.)
            s3 = float32(0.)
            for i in range(CH):
                buf = i & 1
                ks[buf, ty, tx] = rk
                cuda.syncthreads()
                dt = i * TILE
                if i + 1 < CH:  # issue the next load before the math uses this one
                    rk = k[krow, dt + TILE + tx] if valid else float32(0.)
                for j in range(0, TILE, 4):
                    s0 += qs[ty, dt + j] * ks[buf, tx, j]
                    s1 += qs[ty, dt + j + 1] * ks[buf, tx, j + 1]
                    s2 += qs[ty, dt + j + 2] * ks[buf, tx, j + 2]
                    s3 += qs[ty, dt + j + 3] * ks[buf, tx, j + 3]
                # no trailing barrier: chunk i+1 writes the other buffer, and
                # its own barrier orders this buffer's reuse at chunk i+2
            s = ((s0 + s1) + (s2 + s3)) * scale
            if jt + tx >= N:
                s = _NEG_BIG  # masked key: exp() contributes exactly 0

            # --- tile stats via warp shuffles: a warp holds rows ty, ty+1;
            # each row's 16 scores live in one 16-lane half, and xor offsets
            # 8/4/2/1 butterfly-reduce within that half -- no shared memory,
            # no barriers, every lane ends with its row's max and sum
            mt = s
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
            p = math.exp(s - mt)  # P~ uses the *tile* max (FA-1's Algorithm 1)
            lt = p
            lt += cuda.shfl_xor_sync(0xFFFFFFFF, lt, 8)
            lt += cuda.shfl_xor_sync(0xFFFFFFFF, lt, 4)
            lt += cuda.shfl_xor_sync(0xFFFFFFFF, lt, 2)
            lt += cuda.shfl_xor_sync(0xFFFFFFFF, lt, 1)
            ps[ty, tx] = p  # published block-wide by the PV loop's first barrier

            # FA-1 online-softmax merge, unchanged
            mn = mt if mt > m else m
            alpha = math.exp(m - mn)
            beta = math.exp(mt - mn)
            ln = l * alpha + lt * beta
            scale_old = l * alpha / ln  # rescales the already-normalised acc
            scale_new = beta / ln       # normalises this tile's P~ V

            # --- acc = scale_old * acc + scale_new * (P~ @ V), same pipeline
            rv = v[krow, tx] if valid else float32(0.)
            for c in range(CH):
                buf = c & 1
                vs[buf, ty, tx] = rv
                cuda.syncthreads()
                if c + 1 < CH:
                    rv = v[krow, (c + 1) * TILE + tx] if valid else float32(0.)
                p0 = float32(0.)
                p1 = float32(0.)
                p2 = float32(0.)
                p3 = float32(0.)
                for j in range(0, TILE, 4):
                    p0 += ps[ty, j] * vs[buf, j, tx]
                    p1 += ps[ty, j + 1] * vs[buf, j + 1, tx]
                    p2 += ps[ty, j + 2] * vs[buf, j + 2, tx]
                    p3 += ps[ty, j + 3] * vs[buf, j + 3, tx]
                acc[c] = acc[c] * scale_old + ((p0 + p1) + (p2 + p3)) * scale_new
            m = mn
            l = ln

        # acc is already the normalised softmax-weighted sum — write as-is
        if base + ty < N:
            for c in range(CH):
                out[base + ty, c * TILE + tx] = acc[c]

    return flashr


class GpuV3R(GpuPipeline):
    """V3 restructured for latency (register prefetch, split accumulators,
    shuffle stats, one barrier per chunk) — the experiment testing whether
    V3's deficit to V2 is really per-warp latency exposure, not tile sizes."""

    def _attend(self, q, k, v):
        N, D = q.shape
        out = torch.empty((N, D), device=q.device)
        kernel = _flash_reg_kernel(D)
        kernel[math.ceil(N / TILE), (TILE, TILE)](
            q, k, v, np.float32(1.0 / D ** 0.5), out)
        return out
