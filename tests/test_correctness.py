import pytest
import torch
from numba import cuda

from bench import sdpa_reference
from cpu_baseline import CpuPipeline, TorchPipeline
from gpu_v1 import GpuV1
from gpu_v2 import GpuV2
from gpu_v3 import GpuV3

needs_gpu = pytest.mark.skipif(not cuda.is_available(), reason="no CUDA device")


def _check(cls, atol=None, rtol=None):
    torch.manual_seed(0)
    model = cls().eval()
    x = torch.randint(0, 1000, (2, 16))  # [B, N] token ids
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, sdpa_reference(model, x), atol=atol, rtol=rtol)


def test_cpu_matches_torch_sdpa():
    _check(CpuPipeline, atol=1e-4, rtol=1e-3)  # unfused matmul/softmax vs fused SDPA


@needs_gpu
@pytest.mark.parametrize("cls", [GpuV1, GpuV2, GpuV3])
def test_gpu_matches_torch_sdpa(cls):
    _check(cls, atol=1e-4, rtol=1e-3)  # assignment spec: within 1e-4
