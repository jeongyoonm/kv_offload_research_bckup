# Prediction bias and setup reserve

The preceding 8 MiB/depth 2 worker ablation found that 39/42 recorded
compute deadlines fell after A ended. Mean finish prediction error was
approximately +89 ms. B's exposed connector phase sometimes lasted
70–110 ms, including examples without new H2D loads.

This study holds model, TP8 configuration, predictor weights, preparation,
load-only policy, readiness exits, and worker settings constant. It tests:

| Arm | Subtraction from predicted A duration | B setup lead |
| --- | ---: | ---: |
| control | 0 ms | 5 ms |
| bias | 90 ms | 5 ms |
| setup | 0 ms | 100 ms |
| both | 90 ms | 100 ms |

The seal deadline is A's observed start plus its corrected predicted
duration minus B's setup lead. Correction also changes prediction-guided
waiting. The setup lead is a fixed estimate from prior traces, not a
dynamic estimator. Raw and corrected predictions remain separately
observable. Corrected duration is clamped positive. Existing completion,
readiness, cancellation, all-rank preparation, and safety gates remain.

Four counterbalanced repetitions use matching seeds and 75 requests per
arm on the 60K workload, concurrency 32, one output token. That totals
300 measured requests per arm in four separately primed/reset episodes.
Warmups are excluded. All arms retain cached-reference, pinned, compact
preparation and the frozen predictor. The constants are chosen before
this allocation using prior data, not its measured outcomes.

Opt-in worker markers separate deferred store submission from store-flush
waiting. They do not change CUDA stream ordering or block-reuse fences.
The study measures throughput, TTFT, admissions, compute rounds, gaps,
deadline errors, and connector overlap. Earlier sealing can cost batching
opportunities; reduced gaps alone do not establish a throughput win.

```bash
PYTHONPATH="$PWD" .venv/bin/python \
  -m experiments.h2d8_deadline_bias_20261009.run \
  --output /mnt/localnvme/h2d8_deadline_bias_20261009/paired
```
