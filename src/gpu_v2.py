import math

import torch
from numba import cuda, float32

from gpu_base import GpuPipeline

TPB = 128   # threads cooperating on one row in the online softmax

# finite stand-in for -inf: avoids (-inf) - (-inf) = nan when merging two
# partials that saw no elements
_NEG_BIG = float32(-3.0e38)


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
    """V2 — cuBLAS matmuls + a one-pass online softmax kernel.

    Same launch structure as V1; the upgrade that survives library dispatch
    is the softmax: a running (max, sum) pair rescaled when the max moves
    (log-sum-exp trick) gets both statistics out of one coalesced read of
    the row, where V1 pays three. Still materialises the N x N score matrix
    between launches.
    """

    def _step_qkt(self, q, k, scores):
        D = q.shape[1]
        torch.mm(q * D ** -0.5, k.t(), out=scores)

    def _step_softmax(self, scores, weights):
        _softmax_online[scores.shape[0], TPB](scores, weights)

    def _step_weighted_sum(self, weights, v, out):
        torch.mm(weights, v, out=out)
