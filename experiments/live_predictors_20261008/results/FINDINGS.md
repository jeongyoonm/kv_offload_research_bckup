# Findings and verification

All 36 measured cells passed: 2,700 requests, 300 per predictor/workload.
Every request's scheduled token total equals its uncached prompt length.
All cells preserve the frozen checkpoint and matched sealing policy,
use optimized preparation on all ranks, and have no GPU prefix hits or
preemptions. CPU checks passed (211 tests; 8 skipped), and the two new
serving exports match research inference on 5,186 forecasts each
within 0.000001 ms. The owned server stopped; all eight GPUs are free.

## Throughput uncertainty

<!-- markdownlint-disable MD013 -->
| Workload | Predictor | Paired geometric gain | Paired 95% interval |
| --- | --- | ---: | ---: |
| familiar60k | guarded_sparse_mlp | +2.44% | [-7.98%, +14.03%] |
| familiar60k | envelope_sparse_mlp | -2.98% | [-14.48%, +10.07%] |
| short16k | guarded_sparse_mlp | -1.96% | [-14.37%, +12.25%] |
| short16k | envelope_sparse_mlp | -0.66% | [-8.27%, +7.59%] |
| heterogeneous | guarded_sparse_mlp | -0.08% | [-0.34%, +0.17%] |
| heterogeneous | envelope_sparse_mlp | -0.07% | [-0.42%, +0.27%] |
<!-- markdownlint-enable MD013 -->

Intervals use four matched episode log throughput ratios and a
Student-t interval (3 degrees of freedom). Every interval includes
zero; these cells do not establish a throughput improvement or
equivalence. Checkpoint coverage differs as well as architecture.

## Accuracy on identical live snapshots

<!-- markdownlint-disable MD013 -->
| Workload | Predictor | Forecasts | MAE (ms) | MAPE | Bias (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| familiar60k | trees | 92 | 39.56 | 8.31% | -7.44 |
| familiar60k | guarded_sparse_mlp | 92 | 37.30 | 7.68% | -11.04 |
| familiar60k | envelope_sparse_mlp | 92 | 37.69 | 8.43% | +9.27 |
| short16k | trees | 112 | 18.56 | 9.98% | -2.25 |
| short16k | guarded_sparse_mlp | 112 | 30.95 | 16.28% | +5.79 |
| short16k | envelope_sparse_mlp | 112 | 20.71 | 11.77% | +10.29 |
| heterogeneous | trees | 11 | 39.66 | 2.50% | +21.87 |
| heterogeneous | guarded_sparse_mlp | 11 | 70.98 | 5.21% | +50.50 |
| heterogeneous | envelope_sparse_mlp | 11 | 39.46 | 2.93% | +30.83 |
<!-- markdownlint-enable MD013 -->

Positive bias means predicting completion too late. Replay uses
the same 215 causal snapshots for all models; these are sampled by
the union of live policies, rather than an independent held-out
workload distribution. The mixed workload has only 11 forecasts.
The offline 4–5 ms error did not carry over to this live workload.
The MLP modestly improves 60K MAE, but has worse short/mixed MAE.
The blend is close to trees and has a positive completion bias.

## Interpretation

Keep the serving tree as the default. Neither new checkpoint
improves accuracy and throughput consistently across workloads.
For 60K, the MLP's aggregate +2.6% throughput accompanies a reduction
in summed GPU gaps from 1,135 to 464 ms per episode, while compute
rounds increase from 42 to 50 across the four episodes. The blend's
-3.0% accompanies larger gaps (1,566 ms). For 16K, MLP gaps increase
from 89 to 298 ms and aggregate throughput falls 2.0%. These
associations do not isolate prediction error as the causal driver:
changed sealing alters batches and CPU submission timing as well.
Gap totals include load-only intervals, not just post-seal work.

The next targeted study should collect forecasts at the live
scheduler's actual decision times and split evaluation by complete
episodes before fitting. Existing data show forecast timing and
queue-state differences from offline collection; they do not yet
prove a single root cause. Avoid selecting or retuning on these
evaluation labels and then reporting the same cells as validation.

Raw traces, checkpoints and launch metadata:
`/mnt/localnvme/glm53_live_predictors_20261008`.
The benchmark is prefill/TTFT focused: one output token, concurrency
32. It does not establish long-generation throughput or model
output equivalence under the original nondeterministic selector.
