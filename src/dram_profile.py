"""Measured DRAM traffic per attention pass, via Nsight Compute.

The GPU's performance counters are root-gated on stock drivers
(ERR_NVGPUCTRPERM), so this is an offline step, not a notebook cell:

    cd ~/gpt && sudo .venv/bin/python src/dram_profile.py collect

writes dram_profile.json (repo root), which the notebook's bandwidth chart
reads. `collect` shells one ncu run per (version, N); each child process
(`run` mode) JIT-warms the kernels, then wraps exactly one attention pass in
cudaProfilerStart/Stop. ncu captures every kernel launch in that window
(--profile-from-start off) and reports per launch:

    dram__bytes.sum          bytes actually crossing the DRAM<->L2 boundary,
                             refetches included — the real "bytes moved"
    gpu__time_duration.sum   the launch's own duration
    smsp__sass_thread_inst_executed_op_{fadd,fmul,ffma}_pred_on.sum
                             fp32 ops each implementation actually executed
                             (FLOPs = fadd + fmul + 2*ffma; SFU ops like
                             exp/rcp have no per-op counter on Turing)

Summed over the window: bytes/time is the pass's achieved DRAM bandwidth
while kernels were resident, and the FLOP count vs the algorithmic floor
(the two matmuls' 4*N^2*D) is the implementation's compute overhead.

Caveat: ncu locks GPU clocks to base while profiling, so durations run
~2.3x the boost-clock CUDA-event sweep; byte and instruction counts are
clock-independent.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

SEQ_LENS = [2 ** x for x in range(6, 15)]
VERSIONS = ("v1", "v2", "v1n", "v2n", "v3", "sdpa")
METRICS = ",".join([
    "dram__bytes.sum",
    "gpu__time_duration.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
])
OUT_PATH = Path(__file__).resolve().parent.parent / "dram_profile.json"

# ncu spells units both long (nsecond) and short (ns) depending on version
_BYTES = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}
_MS = {"nsecond": 1e-6, "usecond": 1e-3, "msecond": 1.0, "second": 1e3,
       "ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}


def run(version, n):
    """Child mode: one profiled attention pass on device-resident q, k, v."""
    import torch
    import torch.nn.functional as F

    import bench
    from gpu_v1 import GpuV1
    from gpu_v1_numba import GpuV1Numba
    from gpu_v2 import GpuV2
    from gpu_v2_numba import GpuV2Numba
    from gpu_v3 import GpuV3

    model = {"v1": GpuV1, "v2": GpuV2, "v1n": GpuV1Numba, "v2n": GpuV2Numba,
             "v3": GpuV3, "sdpa": GpuV1}[version]()
    q, k, v = (t[0].detach().contiguous().cuda() for t in bench._qkv(model, n))

    def one_pass():
        with torch.no_grad():
            if version == "sdpa":
                F.scaled_dot_product_attention(q, k, v)
            else:
                model._attend(q, k, v)

    one_pass()  # warmup: numba JIT / cuBLAS heuristics, outside the window
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    one_pass()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def _ncu_pass(version, n):
    """Run one profiled pass under ncu; return the summed metric totals."""
    cmd = [shutil.which("ncu") or "/usr/local/cuda/bin/ncu",
           "--profile-from-start", "off", "--metrics", METRICS, "--csv",
           sys.executable, __file__, "run", version, str(n)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    import csv
    import io
    # the CSV header is not necessarily the first stdout line (==PROF== noise)
    lines = r.stdout.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    rows = [] if start is None else \
        [row for row in csv.DictReader(io.StringIO("\n".join(lines[start:])))
         if row.get("Metric Name") in METRICS.split(",")]
    if not rows:  # rc alone is unreliable: ncu can exit nonzero with good data
        sys.exit(f"ncu gave no metrics for {version} N={n}:\n{r.stdout[-500:]}\n{r.stderr[-1500:]}")
    total = dict.fromkeys(METRICS.split(","), 0.0)
    for row in rows:
        name, unit, val = row["Metric Name"], row["Metric Unit"], row["Metric Value"]
        if val in ("n/a", ""):  # metric not collected for this launch
            continue
        scale = _BYTES[unit] if name == "dram__bytes.sum" else \
            _MS[unit] if name == "gpu__time_duration.sum" else 1.0  # inst counts
        total[name] += float(val.replace(",", "")) * scale
    return total


def collect():
    op = "smsp__sass_thread_inst_executed_op_{}_pred_on.sum".format
    out = {}
    for version in VERSIONS:
        out[version] = []
        for n in SEQ_LENS:
            t = _ncu_pass(version, n)
            b, ms = int(t["dram__bytes.sum"]), t["gpu__time_duration.sum"]
            flops = int(t[op("fadd")] + t[op("fmul")] + 2 * t[op("ffma")])
            assert flops > 0, f"FLOP counters came back empty for {version} N={n}"
            out[version].append({"n": n, "dram_bytes": b, "kernel_ms": ms,
                                 "flops": flops})
            print(f"{version:>5} N={n:<5} {b / 1e6:9.1f} MB {flops / 1e9:7.2f} GFLOP "
                  f"in {ms:8.3f} ms", flush=True)
    OUT_PATH.write_text(json.dumps(out, indent=1) + "\n")
    import os
    if os.environ.get("SUDO_UID"):  # hand the artifact back to the invoking user
        os.chown(OUT_PATH, int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]))
    else:
        collect()
