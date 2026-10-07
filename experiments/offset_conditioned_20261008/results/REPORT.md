# Live predictor comparison: GLM 5.3 TP8

Complete: True; measured cells: 9/9.

<!-- markdownlint-disable MD013 -->
| Workload | Predictor | req/s | Throughput gain | Mean TTFT (ms) | p95 TTFT (ms) | Compute rounds | Mean GPU gap/episode (ms) | Issued MAE (ms) | Recovered KV-ready (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| familiar60k | trees | 6.217 | +0.0% | 4431.7 | 6083.7 | 26 | 2408.1 | 43.06 | 100.0 |
| familiar60k | matched_elapsed | 6.159 | -0.9% | 4491.2 | 6027.8 | 27 | 2474.0 | 22.12 | 97.6 |
| familiar60k | matched_offset | 6.430 | +3.4% | 4293.8 | 5512.2 | 29 | 1840.3 | 17.13 | 100.0 |
<!-- markdownlint-enable MD013 -->

Three 75-request episodes per predictor. Each follows a common-tree 32-request
same-workload conditioning episode; throughput excludes priming/startup. Pooled
p95 TTFT across 225 requests. Issued predictions have policy-dependent
populations. Recovery counts distinct late-admitted requests whose KV became
ready during some measured compute interval; it does not prove same-predecessor
recovery. GPU gaps use all-rank start upper bounds and include load-only
intervals.

Matched elapsed/offset models share data and architecture. The control is the
latest broader-data serving tree checkpoint. All policies use the same 5 ms
deadline lead and optimized preparation. The original sparse selector is
preserved. Prediction-cost and actual-seal-lead distributions and paired
confidence intervals are in comparison.json.
