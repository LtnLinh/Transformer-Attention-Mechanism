import math

import numpy as np
from numba import cuda, float32

from gpu_base import GpuPipeline


@cuda.jit
def _matmul(a, b, scale, out):
    """out = scale * (a @ b) — one thread per output element, all reads from
    global memory. The scale rides the epilogue for free: a separate scaling
    launch would re-read and re-write the whole output matrix.

    threadIdx.x walks the output column (the contiguous axis) so warp reads
    of b are coalesced; a[i, kk] is a broadcast within the warp.
    """
    j, i = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        acc = float32(0.)
        for kk in range(a.shape[1]):
            acc += a[i, kk] * b[kk, j]
        out[i, j] = acc * scale


TPB = 256  # threads cooperating on one row


@cuda.jit
def _softmax(x, out):
    """Row softmax, one block per row: max pass, sum pass, write pass.

    Each thread strides over its slice of the row; the block reduces the
    per-thread max and sum in shared memory. Still three reads of the row
    (naive), but the row is now processed cooperatively instead of by a
    single thread.
    """
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n = x.shape[1]

    sm = cuda.shared.array(TPB, float32)
    sd = cuda.shared.array(TPB, float32)

    # row max for numerical stability (avoid exp overflow)
    m = x[row, tid] if tid < n else float32(-3.0e38)
    for j in range(tid + TPB, n, TPB):
        if x[row, j] > m:
            m = x[row, j]
    sm[tid] = m
    cuda.syncthreads()

    stride = TPB // 2
    while stride > 0:
        if tid < stride and sm[tid + stride] > sm[tid]:
            sm[tid] = sm[tid + stride]
        cuda.syncthreads()
        stride //= 2
    m = sm[0]

    d = float32(0.)
    for j in range(tid, n, TPB):
        d += math.exp(x[row, j] - m)
    sd[tid] = d
    cuda.syncthreads()

    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            sd[tid] += sd[tid + stride]
        cuda.syncthreads()
        stride //= 2
    denom = sd[0]

    for j in range(tid, n, TPB):
        out[row, j] = math.exp(x[row, j] - m) / denom


def _grid2d(rows, cols, tpb=(16, 16)):
    # grid x covers columns, grid y covers rows (matches j, i = cuda.grid(2))
    return (math.ceil(cols / tpb[0]), math.ceil(rows / tpb[1])), tpb


class GpuV1Numba(GpuPipeline):
    """The handwritten V1, kept for comparison: naive three-launch attention
    — scaled QK^T matmul, softmax, weighted sum — every kernel written in
    numba, one thread per output element.

    Every intermediate, including the full N x N score matrix, round-trips
    through global memory between kernels, and the softmax reads each row
    three times.
    """

    def _step_qkt(self, q, k, scores):
        N, D = q.shape
        bpg, tpb = _grid2d(N, N)
        # np.float32 keeps the kernel arithmetic in fp32 (a python float is typed float64)
        _matmul[bpg, tpb](q, k.t().contiguous(), np.float32(D ** -0.5), scores)

    def _step_softmax(self, scores, weights):
        _softmax[scores.shape[0], TPB](scores, weights)

    def _step_weighted_sum(self, weights, v, out):
        bpg, tpb = _grid2d(*out.shape)
        _matmul[bpg, tpb](weights, v, np.float32(1.0), out)
