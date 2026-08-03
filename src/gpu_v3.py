import math
from functools import lru_cache

import numpy as np
import torch
from numba import cuda, float32

from gpu_base import GpuPipeline

BR = 16       # query rows per block (Q tile height, one thread row each)
TX = 16       # thread columns per block; each thread's outputs stride by TX
BC = 32       # key/value rows per tile: a 1x2 S micro-tile per thread, so one
              # broadcast Q (or P~) load feeds two FMAs — 1.5 shared loads/FMA
              # instead of the 2.0 a one-element-per-thread mapping pays
DT = 32       # D-chunk staged per step of the two block matmuls
DT_P = 33     # padded chunk width — numba wants a plain constant in
BC_P = 33     # cuda.shared.array, and the pad keeps strides off the 32 banks

# finite stand-in for -inf on masked keys: exp() of it is 0 and it dodges the
# (-inf) - (-inf) = nan trap when a tile holds no real keys
_NEG_BIG = float32(-3.0e38)
_M_INIT = float32(-math.inf)


@lru_cache(maxsize=None)
def _flash_kernel(D):
    """Build the FlashAttention-2 forward kernel for a fixed model width D —
    numba wants compile-time-constant shared/local array shapes."""
    assert D % DT == 0, "flash kernel assumes d_model is a multiple of 32"
    CH = D // TX   # output columns owned by each thread (two per PV d-chunk)
    D_P = D + 1    # padded Q row: plain name, same shared.array constraint

    @cuda.jit
    def flash(q, k, v, scale, out):
        tx = cuda.threadIdx.x          # S columns: keys jt + tx, jt + tx + TX
        ty = cuda.threadIdx.y          # S row: query base + ty
        tid = ty * TX + tx
        base = cuda.blockIdx.x * BR    # first query row of this block
        N = q.shape[0]

        qs = cuda.shared.array((BR, D_P), float32)   # resident Q tile
        ks = cuda.shared.array((BC, DT_P), float32)  # K d-chunk
        vs = cuda.shared.array((BC, DT_P), float32)  # V d-chunk
        ps = cuda.shared.array((BR, BC_P), float32)  # P~ = exp(S - m) tile

        acc = cuda.local.array(CH, float32)  # O[base+ty, tx::TX], unnormalised

        # Stage the block's whole Q tile once (coalesced, all 256 threads); it
        # stays on-chip for the entire K/V sweep — the FlashAttention move
        # that makes Q traffic O(N D) instead of O(N^2 D).
        for idx in range(tid, BR * D, BR * TX):
            r = idx // D
            c = idx - r * D
            qs[r, c] = q[base + r, c] if base + r < N else float32(0.)
        cuda.syncthreads()

        for c in range(CH):
            acc[c] = float32(0.)
        m = _M_INIT      # running row max (all TX lanes of a row track it)
        l = float32(0.)  # running row sum of exponentials

        # staging map, loop-invariant: thread tid always loads column sc_ of
        # rows sr_, sr_+8, sr_+16, sr_+24 of the 32 x 32 chunk (256 threads
        # stride whole 8-row groups — no per-element division in the loop)
        sr = tid // DT
        sc_ = tid - sr * DT

        for jt in range(0, N, BC):
            # S[ty, tx] and S[ty, tx+TX] = scale * dot(q[base+ty], k[jt+.]) —
            # block matmul accumulated over d-chunks; K chunk staged coalesced
            # (4 elements per thread), Q broadcast from the resident tile.
            # The kk loop is unrolled x4 by hand: numba/LLVM leaves 32-trip
            # loops with shared loads rolled, and the loop overhead would
            # otherwise cost more than the two FMAs per trip.
            # per-tile row guards, invariant across d-chunks (V rows == K rows)
            ok0 = jt + sr < N
            ok1 = jt + sr + 8 < N
            ok2 = jt + sr + 16 < N
            ok3 = jt + sr + 24 < N

            s0 = float32(0.)
            s1 = float32(0.)
            # software-pipelined staging: the next chunk is prefetched into
            # registers while the current chunk's FMAs run, so the global-load
            # latency hides behind compute instead of stalling at the barrier
            g0 = k[jt + sr, sc_] if ok0 else float32(0.)
            g1 = k[jt + sr + 8, sc_] if ok1 else float32(0.)
            g2 = k[jt + sr + 16, sc_] if ok2 else float32(0.)
            g3 = k[jt + sr + 24, sc_] if ok3 else float32(0.)
            for dc in range(0, D, DT):
                ks[sr, sc_] = g0
                ks[sr + 8, sc_] = g1
                ks[sr + 16, sc_] = g2
                ks[sr + 24, sc_] = g3
                cuda.syncthreads()
                if dc + DT < D:  # prefetch the next K chunk
                    g0 = k[jt + sr, dc + DT + sc_] if ok0 else float32(0.)
                    g1 = k[jt + sr + 8, dc + DT + sc_] if ok1 else float32(0.)
                    g2 = k[jt + sr + 16, dc + DT + sc_] if ok2 else float32(0.)
                    g3 = k[jt + sr + 24, dc + DT + sc_] if ok3 else float32(0.)
                for kk in range(0, DT, 4):
                    qv = qs[ty, dc + kk]  # one broadcast load, two FMAs
                    s0 += qv * ks[tx, kk]
                    s1 += qv * ks[tx + TX, kk]
                    qv = qs[ty, dc + kk + 1]
                    s0 += qv * ks[tx, kk + 1]
                    s1 += qv * ks[tx + TX, kk + 1]
                    qv = qs[ty, dc + kk + 2]
                    s0 += qv * ks[tx, kk + 2]
                    s1 += qv * ks[tx + TX, kk + 2]
                    qv = qs[ty, dc + kk + 3]
                    s0 += qv * ks[tx, kk + 3]
                    s1 += qv * ks[tx + TX, kk + 3]
                cuda.syncthreads()
            s0 *= scale
            s1 *= scale
            if jt + tx >= N:
                s0 = _NEG_BIG  # masked key: exp() contributes exactly 0
            if jt + tx + TX >= N:
                s1 = _NEG_BIG

            # tile row max: butterfly over the row's TX lanes (never crosses
            # the 16-lane half-warp, so both rows of a warp reduce in step)
            mt = s0
            if s1 > mt:
                mt = s1
            o = cuda.shfl_xor_sync(0xffffffff, mt, 8)
            if o > mt:
                mt = o
            o = cuda.shfl_xor_sync(0xffffffff, mt, 4)
            if o > mt:
                mt = o
            o = cuda.shfl_xor_sync(0xffffffff, mt, 2)
            if o > mt:
                mt = o
            o = cuda.shfl_xor_sync(0xffffffff, mt, 1)
            if o > mt:
                mt = o

            # online softmax update (FA2): one correction per tile, applied to
            # the running sum and the output accumulator only when the max
            # actually moves (uniform across the row's lanes, no divergence
            # hazard — the shuffles stay outside); normalisation is deferred
            # to the very end
            if mt > m:
                corr = math.exp(m - mt)  # first tile: exp(-inf - finite) = 0
                l *= corr
                for c in range(CH):
                    acc[c] *= corr
                m = mt
            p0 = math.exp(s0 - m)
            p1 = math.exp(s1 - m)

            lt = p0 + p1  # tile row sum, same butterfly
            lt += cuda.shfl_xor_sync(0xffffffff, lt, 8)
            lt += cuda.shfl_xor_sync(0xffffffff, lt, 4)
            lt += cuda.shfl_xor_sync(0xffffffff, lt, 2)
            lt += cuda.shfl_xor_sync(0xffffffff, lt, 1)

            l += lt
            ps[ty, tx] = p0
            ps[ty, tx + TX] = p1
            cuda.syncthreads()

            # acc += P~ @ V_tile, V staged in d-chunks: chunk dc covers output
            # columns [dc, dc+DT), and thread (ty, tx) owns columns dc + tx
            # and dc + TX + tx — its acc slots 2*(dc//DT) and 2*(dc//DT) + 1;
            # one broadcast P~ load again feeds two independent FMA chains
            # (j loop hand-unrolled x4 for the same reason as the kk loop)
            g0 = v[jt + sr, sc_] if ok0 else float32(0.)
            g1 = v[jt + sr + 8, sc_] if ok1 else float32(0.)
            g2 = v[jt + sr + 16, sc_] if ok2 else float32(0.)
            g3 = v[jt + sr + 24, sc_] if ok3 else float32(0.)
            c = 0
            for dc in range(0, D, DT):
                vs[sr, sc_] = g0
                vs[sr + 8, sc_] = g1
                vs[sr + 16, sc_] = g2
                vs[sr + 24, sc_] = g3
                cuda.syncthreads()
                if dc + DT < D:  # prefetch the next V chunk
                    g0 = v[jt + sr, dc + DT + sc_] if ok0 else float32(0.)
                    g1 = v[jt + sr + 8, dc + DT + sc_] if ok1 else float32(0.)
                    g2 = v[jt + sr + 16, dc + DT + sc_] if ok2 else float32(0.)
                    g3 = v[jt + sr + 24, dc + DT + sc_] if ok3 else float32(0.)
                a0 = acc[c]
                a1 = acc[c + 1]
                for j in range(0, BC, 4):
                    pj = ps[ty, j]
                    a0 += pj * vs[j, tx]
                    a1 += pj * vs[j, tx + TX]
                    pj = ps[ty, j + 1]
                    a0 += pj * vs[j + 1, tx]
                    a1 += pj * vs[j + 1, tx + TX]
                    pj = ps[ty, j + 2]
                    a0 += pj * vs[j + 2, tx]
                    a1 += pj * vs[j + 2, tx + TX]
                    pj = ps[ty, j + 3]
                    a0 += pj * vs[j + 3, tx]
                    a1 += pj * vs[j + 3, tx + TX]
                acc[c] = a0
                acc[c + 1] = a1
                c += 2
                cuda.syncthreads()

        # deferred normalisation; l > 0 for every real row (its own diagonal
        # key contributes exp(s - m) with m >= s). FA2 would also stash
        # L = m + log(l) for the backward pass — inference-only here.
        if base + ty < N:
            inv = float32(1.) / l
            for c in range(CH):
                out[base + ty, c * TX + tx] = acc[c] * inv

    return flash


class GpuV3(GpuPipeline):
    """V3 — FlashAttention-2 forward, faithfully blocked.

    One thread block per Br = 16 query rows: the Q tile is staged in shared
    memory once and stays resident while the block sweeps the K/V tiles. Per
    tile, the Br x Bc score block S = scale * Q K^T is a shared-memory block
    matmul (each thread owns a 1x2 micro-tile, so every broadcast Q or P~
    load feeds two FMAs); the row max and row sum of the tile come out of
    half-warp butterfly shuffles; the online-softmax correction
    exp(m_old - m_new) rescales the running sum and the output accumulator
    once per tile — and only when the max moves — before P~ V accumulates
    through the same d-chunked staging. Normalisation by the running sum
    happens once at the end (FA2's deferral). The N x N score matrix never
    exists in any memory — extra footprint is O(Br x Bc) per block, and
    Q/K/V are each read once per Q-block sweep, the paper's IO bound.
    """

    def _attend(self, q, k, v):
        N, D = q.shape
        out = torch.empty((N, D), device=q.device)
        kernel = _flash_kernel(D)
        # np.float32 keeps the kernel arithmetic in fp32: a python float
        # scale is typed float64 and would silently promote the hot loops
        kernel[math.ceil(N / BR), (TX, BR)](
            q, k, v, np.float32(1.0 / D ** 0.5), out)
        return out
