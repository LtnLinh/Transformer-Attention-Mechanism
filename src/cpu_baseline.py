import torch
from torch.profiler import record_function

from abstract import TransformerBase


class CpuPipeline(TransformerBase):
    """CPU pipeline: attention as three vectorized torch ops (MKL).

    Puts every stage on the same vectorized (BLAS) footing, so a profile of
    the forward pass reflects each step's algorithmic cost rather than
    interpreter overhead. Kept unfused (matmul, softmax, matmul) so the three
    attention sub-steps stay separately attributable in the profiler.
    """

    def attention(self, q, k, v):
        scale = q.shape[-1] ** -0.5
        with record_function("2a_qk_matmul"):
            scores = q @ k.transpose(-2, -1) * scale
        with record_function("2b_softmax"):
            weights = torch.softmax(scores, dim=-1)
        with record_function("2c_value_weighted_sum"):
            return weights @ v
