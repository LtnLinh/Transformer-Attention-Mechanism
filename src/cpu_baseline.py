import torch
from torch.profiler import record_function

from abstract import TransformerBase


class CpuPipeline(TransformerBase):
    """CPU pipeline: attention as three torch ops, made sequential by
    configuration rather than by hand-written loops — the notebook pins one
    thread (torch.set_num_threads(1)) and caps dispatch at the baseline ISA
    (ATEN_CPU_CAPABILITY=default, MKL_ENABLE_INSTRUCTIONS=SSE2, set before
    torch loads). Library code, sequential execution: profiles reflect each
    step's algorithmic cost with no interpreter overhead and no parallelism.
    Kept unfused (matmul, softmax, matmul) so the three attention sub-steps
    stay separately attributable in the profiler.
    """

    def attention(self, q, k, v):
        scale = q.shape[-1] ** -0.5
        with record_function("2a_qk_matmul"):
            scores = q @ k.transpose(-2, -1) * scale
        with record_function("2b_softmax"):
            weights = torch.softmax(scores, dim=-1)
        with record_function("2c_value_weighted_sum"):
            return weights @ v
