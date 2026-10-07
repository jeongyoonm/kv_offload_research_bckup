# Live predictor comparison: GLM 5.3 TP8

Complete: True; measured cells: 36/36.

<!-- markdownlint-disable MD013 -->
| Workload | Predictor | req/s | Throughput gain | Mean TTFT (ms) | p95 TTFT (ms) | Compute rounds | Mean GPU gap/episode (ms) | Issued MAE (ms) | Recovered KV-ready (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| familiar60k | trees | 6.363 | +0.0% | 4326.1 | 6019.6 | 40 | 1980.5 | 49.86 | 97.9 |
| familiar60k | matched_elapsed | 6.477 | +1.8% | 4265.5 | 6055.9 | 38 | 1777.5 | 18.99 | 99.0 |
| familiar60k | matched_offset | 6.856 | +7.8% | 4026.4 | 5477.8 | 42 | 845.9 | 18.35 | 99.2 |
| short16k | trees | 24.846 | +0.0% | 1116.6 | 1469.1 | 43 | 66.4 | 20.43 | 96.4 |
| short16k | matched_elapsed | 23.018 | -7.4% | 1212.9 | 1523.5 | 41 | 347.4 | 14.38 | 96.1 |
| short16k | matched_offset | 24.872 | +0.1% | 1111.3 | 1482.1 | 43 | 69.4 | 8.09 | 98.9 |
| heterogeneous | trees | 1.722 | +0.0% | 15582.3 | 21417.1 | 100 | 1312.9 | 31.13 | 2.7 |
| heterogeneous | matched_elapsed | 1.722 | +0.0% | 15543.7 | 21430.5 | 101 | 1295.1 | 6.27 | 2.7 |
| heterogeneous | matched_offset | 1.724 | +0.1% | 15583.0 | 21418.4 | 98 | 1285.6 | 7.17 | 2.1 |
<!-- markdownlint-enable MD013 -->

Four 75-request episodes per predictor/workload; throughput excludes
priming/startup. Pooled p95 TTFT across 300 requests. Issued predictions have
policy-dependent populations. Recovery counts distinct late-admitted requests
whose KV became ready during some measured compute interval; it does not prove
same-predecessor recovery. GPU gaps use all-rank start upper bounds and include
load-only intervals.

Matched elapsed/offset models share data and architecture. The control is the
latest broader-data serving tree checkpoint. All policies use the same 5 ms
deadline lead and optimized preparation. The original sparse selector is
preserved. Prediction-cost and actual-seal-lead distributions and paired
confidence intervals are in comparison.json.
