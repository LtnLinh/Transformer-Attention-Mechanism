import math

import torch
from numba import cuda, float32

from gpu_base import GpuPipeline

TPB = 256  # threads cooperating on one row


@cuda.jit
def _softmax(x, out):
    """Row softmax, one block per row: max pass, sum pass, write pass.

    Each thread strides over its slice of the row; the block reduces the
    per-thread max and sum in shared memory. Still three reads of the row
    (naive), but the row is processed cooperatively instead of by a
    single thread.
    """
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n = x.shape[1]

    sm = cuda.shared.array(TPB, float32)
    sd = cuda.shared.array(TPB, float32)

    m = float32(-3.0e38)
    for j in range(tid, n, TPB):
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


class GpuV1(GpuPipeline):
    """V1 — cuBLAS matmuls + a naive three-pass softmax kernel.

    The two matmuls dispatch to the library (torch.mm -> cuBLAS SGEMM), with
    the 1/sqrt(D) scale folded into the cheap O(N D) operand rather than a
    pass over the N x N output. What remains hand-written is the softmax,
    which still reads its row three times (max, sum, write). Every
    intermediate, including the full N x N score matrix, still round-trips
    through global memory between launches.
    """

    def _step_qkt(self, q, k, scores):
        D = q.shape[1]
        torch.mm(q * D ** -0.5, k.t(), out=scores)

    def _step_softmax(self, scores, weights):
        _softmax[scores.shape[0], TPB](scores, weights)

    def _step_weighted_sum(self, weights, v, out):
        torch.mm(weights, v, out=out)
