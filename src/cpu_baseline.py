import math

import torch
from torch.profiler import record_function

from abstract import TransformerBase


def _attend(q, k, v, scale):
    """Attention for one sequence as plain Python lists: [N][D] x3 -> [N][D].

    Both matmuls are the classic naive form — three nested loops, one output
    element per innermost accumulation.
    """
    n, d = len(q), len(q[0])

    with record_function("2a_qk_matmul"):
        scores = [[0.0] * n for _ in range(n)]  # QK^T / sqrt(D)
        for i in range(n):
            for j in range(n):
                s = 0.0
                for x in range(d):
                    s += q[i][x] * k[j][x]
                scores[i][j] = s * scale

    with record_function("2b_softmax"):
        weights = []  # row softmax, max-subtracted for stability
        for row in scores:
            m = max(row)
            exps = [math.exp(s - m) for s in row]
            total = sum(exps)
            weights.append([e / total for e in exps])

    with record_function("2c_value_weighted_sum"):
        out = [[0.0] * d for _ in range(n)]  # weights @ V
        for i in range(n):
            for x in range(d):
                s = 0.0
                for j in range(n):
                    s += weights[i][j] * v[j][x]
                out[i][x] = s
    return out


class CpuPipeline(TransformerBase):
    """CPU pipeline: attention as bare Python loops.

    No BLAS, no SIMD, no threads, no cache blocking — the sequential baseline.
    O(N^2 D) interpreter ops, so the benchmark sweep is capped where its
    runtime stays practical.
    """

    def attention(self, q, k, v):
        scale = q.shape[-1] ** -0.5
        return torch.stack([
            torch.tensor(_attend(q[b].tolist(), k[b].tolist(), v[b].tolist(), scale))
            for b in range(q.shape[0])
        ])
