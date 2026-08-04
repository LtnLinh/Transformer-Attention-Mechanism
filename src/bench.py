import time

import torch
import torch.nn.functional as F
from numba import cuda
from torch.profiler import ProfilerActivity, profile, record_function

STEPS = ["1_embedding", "2a_qk_matmul", "2b_softmax", "2c_value_weighted_sum", "3_ffn"]
ATTN_STEPS = STEPS[1:4]
# coarse variant: attention as one step, so pipelines without the 2a/2b/2c
# sub-step labels (the GPU versions) are comparable
STEPS_TOTAL = ["1_embedding", "2_attention_total", "3_ffn"]


def _tokens(model, n, batch=1):
    return torch.randint(0, model.embedding.num_embeddings, (batch, n))


def _qkv(model, n):
    with torch.no_grad():
        h = model.embedding(_tokens(model, n))
        return model.q_proj(h), model.k_proj(h), model.v_proj(h)


def sdpa_reference(model, x):
    """Full forward pass with attention done by torch SDPA (correctness oracle)."""
    with torch.no_grad():
        h = model.embedding(x)
        attn = F.scaled_dot_product_attention(model.q_proj(h), model.k_proj(h), model.v_proj(h))
        h = model.norm1(h + attn)
        return model.norm2(h + model.ffn(h))


def step_percentages(model, n, warmup=True, steps=STEPS):
    """% of one forward pass spent in each labelled step (torch.profiler).

    Pass warmup=False for pure-Python models: nothing to warm up, and one
    extra forward costs seconds to minutes.
    """
    x = _tokens(model, n)
    with torch.no_grad():
        if warmup:
            model(x)  # first call pays one-time costs (numba JIT / torch thread pool)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("0_total"):
                model(x)

    def t(key):  # a key can appear more than once -> take max over duplicates
        return max((e.cpu_time_total for e in prof.key_averages() if e.key == key), default=0.0)

    total = t("0_total")
    pct = {s: 100.0 * t(s) / total for s in steps}
    pct["other"] = 100.0 - sum(pct.values())
    return pct


def attention_step_ms(model, n):
    """Absolute ms of each attention sub-step, from one profiled pass over
    the record_function labels inside the model's attention. One pass, no
    warmup — meant for the pure-Python CPU model, where a pass costs minutes
    and there is nothing to warm up. Assumes batch=1 (what _qkv provides):
    key_averages sums duplicate labels, so batch>1 would fold all sequences
    into one number."""
    q, k, v = _qkv(model, n)
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            model.attention(q, k, v)

    def t(key):
        return max((e.cpu_time_total for e in prof.key_averages() if e.key == key), default=0.0)

    return {s: t(s) / 1e3 for s in ATTN_STEPS}  # profiler reports microseconds


def attention_ms(model, n, reps=5, warmup=True):
    """Average wall time (ms) of one attention step on precomputed q, k, v.

    GPU pipelines synchronize and copy back to CPU inside `attention`, so
    a wall clock around the call is correct and includes the transfer cost.
    Pass warmup=False for pure-Python models: nothing to JIT, and one extra
    pass costs seconds to minutes.
    """
    q, k, v = _qkv(model, n)
    with torch.no_grad():
        if warmup:
            model.attention(q, k, v)  # warmup / JIT
        t0 = time.perf_counter()
        for _ in range(reps):
            model.attention(q, k, v)
    return (time.perf_counter() - t0) / reps * 1e3


def sdpa_gpu_ms(model, n, reps=5):
    """torch SDPA on GPU, same contract as `attention_ms`: CPU tensors in,
    CPU result out, with both transfers inside the timed region."""
    q, k, v = _qkv(model, n)
    with torch.no_grad():
        F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu()  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def gpu_specs():
    """Peak fp32 TFLOP/s and memory bandwidth, HW02 roofline style: device
    query for SMs/cc, pynvml for clocks, known-family tables for bus width
    and cores per SM — with a per-family fallback when pynvml is missing."""
    gpu = cuda.get_current_device()
    cc = tuple(gpu.compute_capability)
    sm = gpu.MULTIPROCESSOR_COUNT
    name = gpu.name.decode() if isinstance(gpu.name, bytes) else gpu.name
    cores_per_sm = {(7, 0): 64, (7, 5): 64, (8, 0): 64,
                    (8, 6): 128, (8, 9): 128, (9, 0): 128}.get(cc, 128)
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        sm_mhz = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
        mem_mhz = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)
        try:
            bus_bits = pynvml.nvmlDeviceGetMemoryBusWidth(h)
        except pynvml.NVMLError:  # per-family fallback
            bus_bits = {(7, 0): 4096, (7, 5): 256, (8, 0): 5120,
                        (8, 6): 384, (8, 9): 384, (9, 0): 6144}.get(cc, 256)
        bw_gbs = mem_mhz * 1e6 * bus_bits / 8 * 2 / 1e9  # DDR: 2 transfers/clock
        pynvml.nvmlShutdown()
    except Exception:  # no pynvml: boost clock + bandwidth from spec sheets
        sm_mhz, bw_gbs = {(7, 5): (1590, 300.0), (8, 0): (1410, 1555.0),
                          (9, 0): (1785, 3900.0)}.get(cc, (1500, 500.0))
    tflops = sm * cores_per_sm * 2 * sm_mhz * 1e6 / 1e12
    return {"name": name, "sm": sm, "tflops": tflops, "bw_gbs": bw_gbs}


def attention_kernel_ms(model, n, reps=5):
    """Device-resident attention time via CUDA events (HW02's cuda_time_ms):
    q/k/v already on the GPU, no transfers in the timed region."""
    q, k, v = (t[0].detach().contiguous().cuda() for t in _qkv(model, n))
    model._attend(q, k, v)  # warmup / JIT
    cuda.synchronize()
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    start.record()
    for _ in range(reps):
        model._attend(q, k, v)
    end.record()
    end.synchronize()
    return cuda.event_elapsed_time(start, end) / reps


def attention_kernel_step_ms(model, n, reps=5):
    """Device-resident time (ms) of each attention step via CUDA events.

    Runs the steps in pipeline order so each timed step reads real data
    left behind by the previous one."""
    q, k, v = (t[0].detach().contiguous().cuda() for t in _qkv(model, n))
    N, D = q.shape
    scores = torch.empty((N, N), device=q.device)
    weights = torch.empty((N, N), device=q.device)
    out = torch.empty((N, D), device=q.device)
    steps = {"2a_qk_matmul": lambda: model._step_qkt(q, k, scores),
             "2b_softmax": lambda: model._step_softmax(scores, weights),
             "2c_value_weighted_sum": lambda: model._step_weighted_sum(weights, v, out)}

    times = {}
    for name, fn in steps.items():
        fn()  # warmup / JIT; also fills the buffer the next step reads
        cuda.synchronize()
        start = cuda.event(timing=True)
        end = cuda.event(timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        end.synchronize()
        times[name] = cuda.event_elapsed_time(start, end) / reps
    return times


def sdpa_kernel_ms(model, n, reps=5):
    """Device-resident torch SDPA time via CUDA events."""
    q, k, v = (t.detach().cuda() for t in _qkv(model, n))
    with torch.no_grad():
        F.scaled_dot_product_attention(q, k, v)  # warmup
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            F.scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def attention_flops(model, n):
    """FLOP count of one [N, D] attention, from torch's own flop_counter
    (torch.utils.flop_counter) — the per-matmul formula is torch's, not a
    hand-rolled 4*N^2*D. Counts the two matmuls the kernels actually do:
    scores = Q @ K^T and out = weights @ V, each 2*N^2*D FLOPs. The softmax's
    elementwise ops are not FLOP-counted (standard roofline convention). The
    math is identical for V1/V2/V3, so one count serves every version.

    Runs on CPU — counting is analytic, it just needs the ops to dispatch."""
    from torch.utils.flop_counter import FlopCounterMode
    d = model.embedding.embedding_dim
    q = torch.zeros(1, n, d)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        w = torch.bmm(q, q.transpose(-1, -2))  # Q @ K^T  -> [1, N, N]
        torch.bmm(w.softmax(-1), q)            # weights @ V -> [1, N, D]
    return fc.get_total_flops()


def attention_bytes(model, n):
    """Compulsory DRAM traffic (bytes) of one [N, D] attention: read Q, K, V
    and write the output, each element touched once. Byte width comes from the
    tensor (element_size), not a magic constant. This is the roofline's
    lower-bound traffic; real kernels move more — V1 re-reads its operands per
    output element, V2/V3 far less.
    ponytail: compulsory-traffic model, so V1/V2/V3 share one x-position and
    differ only in how close to the ceiling they land. Measured DRAM bytes —
    which expose the reuse gap directly — come from src/dram_profile.py
    (Nsight Compute dram__bytes.sum; root-gated counters)."""
    probe = torch.empty(n, model.embedding.embedding_dim)
    return 4 * probe.numel() * probe.element_size()  # Q, K, V in + out out


def roofline(model, seq_lens, kernel_ms, specs=None):
    """Operating points for a roofline: achieved TFLOP/s and arithmetic
    intensity (FLOP/byte) per version per N, from library FLOPs / bytes and
    the already-measured kernel-only times. `kernel_ms` is the ktimes dict
    (version -> [ms per n]). Also returns achieved GB/s and % of peak."""
    specs = specs or gpu_specs()
    pts = {}
    for name, ms_list in kernel_ms.items():
        rows = []
        for n, ms in zip(seq_lens, ms_list):
            f, b, t = attention_flops(model, n), attention_bytes(model, n), ms / 1e3
            tflops = f / t / 1e12
            rows.append({"n": n, "ai": f / b, "tflops": tflops, "gbs": b / t / 1e9,
                         "pct_peak": 100.0 * tflops / specs["tflops"]})
        pts[name] = rows
    return pts, specs


def bench_attention(classes, seq_lens, reps=5):
    """Attention-step time for each pipeline class at each sequence length."""
    times = {name: [] for name in classes}
    for n in seq_lens:
        for name, cls in classes.items():
            model = cls().eval()
            times[name].append(attention_ms(model, n, reps))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # drop cached N x N blocks between runs
    return times
