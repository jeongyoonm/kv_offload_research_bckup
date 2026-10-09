# H2D calibration with compute submission interference

Branch: `experiment/h2d-compute-interference-calibration-20261009`.
Source: `/mnt/localnvme/vllm_h2d_interference_20261009`.
Artifacts: `/mnt/localnvme/h2d_interference_20261009`.
AI assistance: Codex.

Use unchanged GLM 5.3 NVFP4 TP8, FP8 KV, the original sparse selector,
96 GiB GPU KV per rank, NUMA placement and process-local THP disabling.
No model weights, native copy code, scheduler policy or serving source changes.
The calibration uses offline fixed eight-request cohorts with GPU-resident
prefixes and unrelated native KV-copy buffers in the same model worker process.
The ordinary CPU offloading connector is omitted to exclude incidental stores
and restorations. Private CPU mmap initialization is reused for copy buffers.

Screen chunks 8/16/32/64/128 MiB and depths 1/2. Every transfer trial queues four
FIFO loads per rank, preserving the 4,096-descriptor limit, with restored prefix
volumes of 4,096, 15,552 and 58,368 tokens. Compute cohorts have contexts of those
same lengths and respectively 64, 512 and 2,048 uncached tokens per request.
The full screen has 150 copy-only controls, 15 compute-only controls and 450
concurrent cells at five randomized repeats. Confirmation also starts copies
after compute has been enqueued. This diagnostic changes overlap timing and is
not a pure subtraction isolating driver contention. Its ready fence uses a
dedicated idle submission stream: these private copy buffers have no dependency
on model work. Otherwise the native submitter's caller-stream fence would force
the copies to wait for the previously enqueued forward. The concurrent screen
keeps the ordinary caller-stream fence, matching production submission order.

Compute priming, shape warmup, native warmup, descriptor allocation and validation
are untimed. Native compute observation brackets the model forward and its small
output-state bookkeeping, excluding input preparation and sampling. Host launch
time, native GPU start/end intervals, GPU event span, FIFO copy completion stamps
and sample bytes are saved on every rank. All ranks drain before validation.
Each measured cell must be one forward with exact context and suffix lengths.
Pause before enqueuing each cohort and resume after all requests arrive, keeping
async scheduling enabled. This prevents frontend enqueue timing from splitting
the intended batch. Use the serving generator's tokenized random KV-number text.
Suffix tails are matched within repeat except for a unique first token that
prevents GPU suffix-cache hits. This token can change model routing, so random
order and repeats are required; this is not a bit-identical workload replay.

Selection considers joint compute/copy completion and compute submission cost,
not transfer bandwidth alone. Retain the current 128 MiB/depth 2 control.
Report the end-timestamp uncertainty and distinguish physical span from kernel
busy time. The calibration is not a serving throughput or model-accuracy test.
Shortlisted settings require confirmation followed by serving validation.

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest \
  experiments/h2d_interference_20261009/verify.py -q
PYTHONPATH="$PWD" .venv/bin/python \
  experiments/h2d_interference_20261009/run.py \
  --output /mnt/localnvme/h2d_interference_20261009/pilot \
  --pilot --confirmation --repeats 1
```

Use a fresh output path. Launch under an independent timeout and retain process
ownership. GPU occupancy is checked before loading; normal/error paths shut down
the owned executor. Preserve rejected cells and tracebacks rather than silently
including them in summaries. No automatic startup selection is implemented yet.

## Recorded stages and analysis

The five-repeat screen fixes confirmation nominees before its independent
eight-repeat allocation: 128/2 (control), 8/2 (joint completion nominee), and
16/2 (readiness compromise). Analyze each completed allocation with `analyze.py`,
and use `select.py` on the screen only. `plot.py` exports joint completion,
forward enqueue and first-load readiness panels. Use the isolated plotting
environment under `/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv`.

The bounded Nsight diagnostic uses `run.py --trace --profiles 60k --shapes short
--configs 128:2 8:2 16:2 --jobs 1 --repeats 1`. Start Nsight with CUDA and NVTX
tracing, no CPU sampling, and `--capture-range=cudaProfilerApi
--capture-range-end=stop`. Set the profiler's `TMPDIR` under the artifact root
before launch and enable `VLLM_KV_READINESS_GPU_TRACE=1` for native copy ranges.
Export to SQLite and run `trace_report.py` on the diagnostic directory.
The one-load profiled cells have a different load count and profiler overhead;
exclude them from selection. Copy API overlap with long compute launch gaps
does not identify the specific internal lock. Recorded kernel time includes
NCCL waiting kernels and does not measure useful compute utilization.

`serving.py --chunk-mib N --depth 2 --output FRESH_PATH` runs the latest
preparation and scheduling prototype with its frozen predictor, deadline,
readiness exits, load-only target, and efficient waiting unchanged. Each setting
has a separate allocation, three excluded warmups, and two matched 32-request
episodes on each of 16K cached, 60K cached and heterogeneous workloads.
Cached-token accounting, no GPU prefix-cache hits/preemptions, one output token,
frozen predictor state and all-rank compact/pinned preparation are required.
The existing snapshot is trained at 128 MiB. For 8/16 MiB, explicitly retarget
only its `context.h2d_chunk_bytes` compatibility identity; preserve every learned
weight and history, and leave the production compatibility check active.
Record the original file hash, loaded state, and changed identity field. A CPU
replay checks exact prediction equality on control request-shape events. This
does not assume identical live prediction error under changed I/O interference.
Run `report.py` on the artifact root after all serving stages, then
`plot_serving.py` using the plotting environment. These small reset/primed
episodes have allocation/order confounds and do not replace a continuous
300-request evaluation or establish model accuracy.
