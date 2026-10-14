# Live predictor comparison: GLM 5.3 TP8

Complete: True; measured cells: 36/36.

<!-- markdownlint-disable MD013 -->
| Workload | Predictor | req/s | Throughput gain | Mean TTFT (ms) | p95 TTFT (ms) | Compute rounds | Mean GPU gap/episode (ms) | Issued MAE (ms) | Recovered KV-ready (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| familiar60k | trees | 6.709 | +0.0% | 4132.5 | 5503.2 | 42 | 1135.0 | 47.76 | 97.0 |
| familiar60k | guarded_sparse_mlp | 6.884 | +2.6% | 4016.2 | 5329.4 | 50 | 463.7 | 34.75 | 94.8 |
| familiar60k | envelope_sparse_mlp | 6.510 | -3.0% | 4228.8 | 5474.6 | 42 | 1565.6 | 38.95 | 98.6 |
| short16k | trees | 24.618 | +0.0% | 1122.3 | 1458.0 | 44 | 89.5 | 18.26 | 96.6 |
| short16k | guarded_sparse_mlp | 24.114 | -2.0% | 1143.9 | 1555.5 | 42 | 297.9 | 29.61 | 99.6 |
| short16k | envelope_sparse_mlp | 24.467 | -0.6% | 1134.4 | 1463.5 | 43 | 73.5 | 19.30 | 97.8 |
| heterogeneous | trees | 1.722 | +0.0% | 15585.8 | 21419.5 | 99 | 1312.3 | 39.33 | 1.3 |
| heterogeneous | guarded_sparse_mlp | 1.721 | -0.1% | 15588.0 | 21415.4 | 98 | 1342.5 | 82.99 | 2.7 |
| heterogeneous | envelope_sparse_mlp | 1.721 | -0.1% | 15580.0 | 21423.1 | 98 | 1341.6 | 33.34 | 3.4 |
<!-- markdownlint-enable MD013 -->

Four 75-request episodes per predictor/workload; throughput excludes
priming/startup. Pooled p95 TTFT across 300 requests. Issued predictions have
policy-dependent populations. Recovery counts distinct late-admitted requests
whose KV became ready during some measured compute interval; it does not prove
same-predecessor recovery. GPU gaps use all-rank start upper bounds and include
load-only intervals.

Checkpoint coverage differs as well as architecture. The control is the latest
broader-data serving tree checkpoint. All policies use the same 5 ms deadline
lead and optimized preparation. The original sparse selector is preserved.
Prediction-cost and actual-seal-lead distributions and paired confidence
intervals are in comparison.json.
