# GPU-only prefix-cache references

The GPU-only allocation completed and stopped. It preserved GLM 5.3 NVFP4,
TP8, original sparse selection, FP8 KV and the fixed 96 GiB/rank GPU pool.
Warm priming and startup are excluded from all serving measurements.

| Prompt / working set | Count | Req/s | Mean TTFT | p95 TTFT |
| --- | ---: | ---: | ---: | ---: |
| 60K, 32 resident | 320 | 7.674 | 3076.8 ms | 4150.3 ms |
| 16K, 75 resident | 300 | 25.797 | 1091.6 ms | 1285.0 ms |
| 60K, 75 over capacity | 300 | 0.494 | 51335.2 ms | 64645.8 ms |

Both resident cases restored 100% of their expected prefix tokens from GPU
cache, with zero CPU transfers and no preemptions. The 75-prefix 60K working
set exceeds the unchanged GPU capacity; all four primed episodes had zero
GPU prefix hits. This is an explicit capacity control, not an all-hit ceiling.

The 60K resident case comprises ten independently primed 32-request waves;
the others comprise four independently primed 75-request episodes. TTFT
percentiles pool measured requests. These are measured references with
ordinary GPU scheduling and preparation, not mathematical upper bounds.
Matched CPU restoration controls are scheduled with the identical shapes,
working sets, seeds and output lengths; their results will be added separately.

Raw logs, request results, launch configuration, cache counters and verified
archive member hashes are retained under `artifacts/gpu_only`.
