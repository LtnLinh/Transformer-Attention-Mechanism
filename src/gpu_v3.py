import math
from functools import lru_cache

import numpy as np
import torch
from numba import cuda, float32

from gpu_base import GpuPipeline

TILE = 16      # tile edge: 16 query rows x 16 key rows per block pair
TILE_P = 17    # padded row width for the score tile (dodges bank conflicts)

# finite stand-in for -inf on masked keys: exp() of it is 0 and it dodges the
# (-inf) - (-inf) = nan trap when a tile holds no real keys
_NEG_BIG = float32(-3.0e38)
_M_INIT = float32(-math.inf)


@lru_cache(maxsize=None)
def _flash_kernel(D):
    """Build the FlashAttention-1 forward kernel for a fixed model width D —
    numba wants compile-time-constant shared/local array shapes.

    The tiling is the paper's standard layout: the Q, K and V blocks are all
    full-width [TILE, D] tiles resident in shared memory together — at
    d_model = 128 in fp32 that is ~26 KB of the 48 KB static budget, the
    regime (d^2 << SRAM) every FlashAttention figure assumes. Each K/V block
    is staged whole, once, and consumed from on-chip memory."""
    assert D % TILE == 0, "flash kernel assumes d_model is a multiple of 16"
    CH = D // TILE  # output columns owned by each thread (cols tx, tx+16, ...)
    D_P = D + 1     # padded row width: shared.array wants plain constants,
    #                 and the pad keeps column-wise reads off one bank
    # Q, K, V full-width tiles + the score tile must fit static shared memory
    assert (3 * TILE * D_P + TILE * TILE_P) * 4 <= 49152, \
        "full-width tiles need d_model <= 254; chunk the staging beyond that"

    @cuda.jit
    def flash(q, k, v, scale, out):
        tx = cuda.threadIdx.x          # key row within the tile
        ty = cuda.threadIdx.y          # query row within the tile
        base = cuda.blockIdx.x * TILE  # first query row of this block
        N = q.shape[0]

        qs = cuda.shared.array((TILE, D_P), float32)     # resident Q tile
        ks = cuda.shared.array((TILE, D_P), float32)     # K block, full width
        vs = cuda.shared.array((TILE, D_P), float32)     # V block, full width
        ps = cuda.shared.array((TILE, TILE_P), float32)  # S, then P~ = exp(S - m~)

        # Stage the block's Q tile once (each thread strides one row); it
        # stays on-chip for the whole K/V sweep — the FlashAttention move
        # that makes Q traffic O(N D) instead of O(N^2 D).
        for c in range(tx, D, TILE):
            qs[ty, c] = q[base + ty, c] if base + ty < N else float32(0.)
        # no barrier needed here: the first staging barrier below covers it

        acc = cuda.local.array(CH, float32)  # O[base+ty, tx::TILE] — kept
        for c in range(CH):                  # normalised after every tile (FA-1)
            acc[c] = float32(0.)
        m = _M_INIT      # running row max
        l = float32(0.)  # running row sum of exponentials

        for jt in range(0, N, TILE):
            # Stage this sweep step's whole [TILE, D] K and V blocks — the
            # standard all-resident layout (each thread strides one row).
            for c in range(tx, D, TILE):
                ks[ty, c] = k[jt + ty, c] if jt + ty < N else float32(0.)
                vs[ty, c] = v[jt + ty, c] if jt + ty < N else float32(0.)
            cuda.syncthreads()

            # S[ty, tx] = scale * dot(q[base+ty], k[jt+tx]), all from shared
            s = float32(0.)
            for kk in range(D):
                s += qs[ty, kk] * ks[tx, kk]
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

            # acc = scale_old * acc + scale_new * (P~ @ V_tile), V read from
            # the resident block
            for c in range(CH):
                pv = float32(0.)
                for j in range(TILE):
                    pv += ps[ty, j] * vs[j, c * TILE + tx]
                acc[c] = acc[c] * scale_old + pv * scale_new
            m = mn
            l = ln
            cuda.syncthreads()  # everyone done with ks/vs/ps before restaging

        # acc is already the normalised softmax-weighted sum — write as-is
        if base + ty < N:
            for c in range(CH):
                out[base + ty, c * TILE + tx] = acc[c]

    return flash


class GpuV3(GpuPipeline):
    """V3 — FlashAttention-1 forward: tiled + online softmax + one fused
    kernel, and the N x N score matrix never exists in any memory.

    One thread block per 16 query rows: the Q tile is staged in shared memory
    once and stays resident while the block sweeps the K/V blocks; each sweep
    step stages one whole [16, D] K block and one whole [16, D] V block beside
    it — the paper's standard all-resident layout, which fits on-chip because
    d_model = 128 keeps d^2 well under the SRAM size. Per 16x16 tile the
    block computes the score tile, takes its row max and sum, and applies
    Algorithm 1's update: the running (max, sum) statistics merge with the
    tile's and the output block is rescaled and renormalised immediately, so
    it always holds the exact softmax-weighted sum of every key seen so far.
    No micro-tiles, no shuffles, no prefetching. Extra footprint is one
    [N, D] output — O(N) — and Q is read once, K/V once per Q-block sweep.
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
