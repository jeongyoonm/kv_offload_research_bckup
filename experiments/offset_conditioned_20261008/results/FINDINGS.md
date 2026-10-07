# Conditioned predictor comparison: GLM 5.3 TP8

Each measured episode follows an excluded common-tree 32-request
episode using the same workload and seed. Measured caches are then
reset and primed separately. Three balanced rotations use fresh
seeds 3301–3303. Checkpoints remain frozen from the primary study.
Each predictor has 225 measured requests, for 675 in total.

<!-- markdownlint-disable MD013 -->
| Predictor | req/s | Mean TTFT (ms) | p95 TTFT (ms) |
| --- | ---: | ---: | ---: |
| trees | 6.217 | 4431.7 | 6083.7 |
| matched_elapsed | 6.159 | 4491.2 | 6027.8 |
| matched_offset | 6.430 | 4293.8 | 5512.2 |
<!-- markdownlint-enable MD013 -->

Paired offset-only versus elapsed-input gain: +4.31%
with a 95% Student-t interval [-33.23%, +62.96%].
Three episodes provide limited precision; this is not equivalence.

## Conditioning phase check

The conditioning checkpoint and procedure are identical across models.
Its throughput is measured on 32 excluded requests, while each main
episode uses 75; the absolute values are not equal-length comparisons.
Variation among conditioning episodes exposes phase differences even
without a predictor change. Conditioning does not guarantee identical
physical state at the start of each measured episode.

<!-- markdownlint-disable MD013 -->
| Repeat | Measured predictor | Common-tree conditioning req/s | Measured req/s |
| ---: | --- | ---: | ---: |
| 0 | matched_elapsed | 5.882 | 5.619 |
| 0 | matched_offset | 6.809 | 7.134 |
| 0 | trees | 5.752 | 5.574 |
| 1 | matched_elapsed | 5.915 | 6.113 |
| 1 | matched_offset | 6.559 | 6.122 |
| 1 | trees | 6.836 | 7.159 |
| 2 | matched_elapsed | 6.926 | 6.872 |
| 2 | matched_offset | 5.928 | 6.134 |
| 2 | trees | 5.917 | 6.119 |
<!-- markdownlint-enable MD013 -->

## Accuracy at common live snapshots

<!-- markdownlint-disable MD013 -->
| Predictor | Points | MAE (ms) | MAPE | Bias (ms) |
| --- | ---: | ---: | ---: | ---: |
| trees | 45 | 45.82 | 10.09% | -12.14 |
| matched_elapsed | 45 | 20.13 | 3.20% | -9.02 |
| matched_offset | 45 | 16.98 | 2.36% | -8.56 |
<!-- markdownlint-enable MD013 -->

All issued forecasts, feature snapshots, and deadlines replay exactly.
All measured requests have exact uncached-token accounting and
unique ordered completion steps. Frozen models, unchanged policy,
all-rank compact/pinned preparation, CPU restoration, zero GPU prefix
hits and zero preemptions are verified. GPU servers are stopped.

This controls the immediately preceding workload episode; it does
not identify the physical cause of any phase effect. No fresh live
labels were used for training, selection, or checkpoint changes.
Raw traces and reports: `/mnt/localnvme/glm53_offset_conditioned_20261008`.
