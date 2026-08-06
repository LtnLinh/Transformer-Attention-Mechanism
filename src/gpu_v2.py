import math

import numpy as np
from numba import cuda, float32

from gpu_base import GpuPipeline

TILE = 16          # tile edge for the shared-memory matmuls
TILE_P = TILE + 1  # padded tile width: numba wants a plain constant in
                   # cuda.shared.array, and the pad avoids bank conflicts
TPB = 128   # threads cooperating on one row in the online softmax

# finite stand-in for -inf: avoids (-inf) - (-inf) = nan when merging two
# partials that saw no elements
_NEG_BIG = float32(-3.0e38)


@cuda.jit
def _qkt_tiled(q, k, scale, out):
    """out[i, j] = scale * dot(q[i], k[j]) — QK^T with both operand tiles
    staged in shared memory; the 1/sqrt(D) scaling is fused into the epilogue.

    threadIdx.x walks the contiguous axis of q and k, so all global loads are
    coalesced; the +1 column pad keeps the sk reads free of bank conflicts.
    """
    sq = cuda.shared.array((TILE, TILE_P), float32)
    sk = cuda.shared.array((TILE, TILE_P), float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = cuda.blockIdx.x * TILE + tx   # output column
    i = cuda.blockIdx.y * TILE + ty   # output row
    jr = cuda.blockIdx.x * TILE + ty  # k row staged by this thread
    N, D = q.shape

    acc = float32(0.)
    for t in range(0, D, TILE):
        sq[ty, tx] = q[i, t + tx] if i < N and t + tx < D else float32(0.)
        sk[ty, tx] = k[jr, t + tx] if jr < N and t + tx < D else float32(0.)
        cuda.syncthreads()
        for kk in range(TILE):
            acc += sq[ty, kk] * sk[tx, kk]
        cuda.syncthreads()

    if i < N and j < N:
        out[i, j] = acc * scale


@cuda.jit
def _matmul_tiled(a, b, out):
    """out = a @ b with shared-memory tiling (used for weights @ V)."""
    sa = cuda.shared.array((TILE, TILE_P), float32)
    sb = cuda.shared.array((TILE, TILE_P), float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = cuda.blockIdx.x * TILE + tx   # output column
    i = cuda.blockIdx.y * TILE + ty   # output row
    M, K = a.shape
    P = b.shape[1]

    acc = float32(0.)
    for t in range(0, K, TILE):
        sa[ty, tx] = a[i, t + tx] if i < M and t + tx < K else float32(0.)
        sb[ty, tx] = b[t + ty, j] if t + ty < K and j < P else float32(0.)
        cuda.syncthreads()
        for kk in range(TILE):
            acc += sa[ty, kk] * sb[kk, tx]
        cuda.syncthreads()

    if i < M and j < P:
        out[i, j] = acc


@cuda.jit
def _softmax_online(x, out):
    """Row softmax in two reads of the row instead of three.

    One block per row. Each thread keeps a running (max, sum) over its slice,
    rescaling the sum whenever the max moves (log-sum-exp trick), so max and
    sum come out of a single pass. Partials merge in shared memory, then a
    second pass normalises and writes. All accesses are coalesced (adjacent
    threads read adjacent columns).
    """
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n = x.shape[1]

    sm = cuda.shared.array(TPB, float32)
    sl = cuda.shared.array(TPB, float32)

    m = _NEG_BIG
    l = float32(0.)
    for j in range(tid, n, TPB):
        val = x[row, j]
        if val > m:
            l *= math.exp(m - val)  # rescale the sum accumulated so far
            m = val
        l += math.exp(val - m)
    sm[tid] = m
    sl[tid] = l
    cuda.syncthreads()

    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            m2 = sm[tid + stride]
            l2 = sl[tid + stride]
            if m2 > sm[tid]:
                sl[tid] = sl[tid] * math.exp(sm[tid] - m2) + l2
                sm[tid] = m2
            else:
                sl[tid] += l2 * math.exp(m2 - sm[tid])
        cuda.syncthreads()
        stride //= 2

    m = sm[0]
    l = sl[0]
    for j in range(tid, n, TPB):
        out[row, j] = math.exp(x[row, j] - m) / l


class GpuV2(GpuPipeline):
    """V2 — tiled QK^T in shared memory + online softmax.

    Still materialises the N x N score matrix, but each element of Q and K is
    read from global memory TILE (16x) times less, and the softmax saves one
    full read of the score matrix.
    """

    def _step_qkt(self, q, k, scores):
        N, D = q.shape
        bpg = (math.ceil(N / TILE), math.ceil(N / TILE))
        # np.float32 keeps the kernel arithmetic in fp32 (a python float is typed float64)
        _qkt_tiled[bpg, (TILE, TILE)](q, k, np.float32(1.0 / D ** 0.5), scores)

    def _step_softmax(self, scores, weights):
        _softmax_online[scores.shape[0], TPB](scores, weights)

    def _step_weighted_sum(self, weights, v, out):
        N, D = out.shape
        bpg = (math.ceil(D / TILE), math.ceil(N / TILE))  # (cols, rows)
        _matmul_tiled[bpg, (TILE, TILE)](weights, v, out)
