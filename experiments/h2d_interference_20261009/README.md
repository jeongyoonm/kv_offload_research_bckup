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
not a pure subtraction isolating driver contention.

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
