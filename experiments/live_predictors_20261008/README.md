# Live predictor comparison: GLM 5.3 TP8

Implementation branch: `experiment/glm53-live-predictor-comparison-20261008`.
Artifacts: `/mnt/localnvme/glm53_live_predictors_20261008`.
Parent: `70bce4c`, with compact cached/pinned preparation, private CPU KV
initialization, async chunked H2D and predictive sealing.

Compare the latest serving tree checkpoint from the dataset coverage study,
guarded sparse MLP, and the 50/50 guarded sparse MLP / guarded envelope with
boosted-tree correction. Frozen seed-73 exports are reused; no online fitting.
The snapshots share their legacy fallback and preparation histories. New
predictors have broader fitting data than the serving tree control, so this
measures predictor choice rather than a pure architecture-only ablation.

Keep the original model selector, TP8, FP8 KV, 64K limit, 96 GiB GPU KV per rank,
2.13 TB aggregate CPU cache, process-local THP disabling, and NUMA placement.
All policies use 16 slots, a 5 ms deadline lead, readiness exits, 2 ms fine
polling, prediction-guided waits, coalesced notifications, and 2 s safety timeout.
No diagnostic selectors or DMA auditing run during throughput measurements.

The independent CPU check compares live feature extraction and predictions
with research implementations on 5,186 recorded forecast rows. Partial/future
rank snapshots decline predictions, checkpoint normalization is validated, and
estimator snapshots round-trip without training libraries in serving.

Each workload receives four rotating-order matched 75-request episodes at
concurrency 32: 300 requests per predictor and workload. Request seeds are
identical within a repeat. Shape and policy warmups, priming, and resets are
excluded. Workloads are 60K/95%-cached, 16K/95%-cached, and heterogeneous
context/suffix/CPU cache-hit mixtures. This is not a continuous 300-request run.
All-rank preparation coverage, zero GPU-prefix hits/preemptions, CPU restoration,
and unchanged exported predictor state are required for every accepted cell.

Run CPU equivalence using the existing CPU evaluation environment:

```bash
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/live_predictors_20261008/verify.py
```

Run serving from this checkout:

```bash
PYTHONPATH="$PWD" .venv/bin/python experiments/live_predictors_20261008/run.py
```

GPU ownership preflight rejects occupied devices. A four-hour shutdown timer
bounds allocation. Each result is persisted immediately. Normal/failure exits
stop the owned process group and save incomplete results. Port: 18408.

## Completed comparison

All 36 cells passed. Results and interpretation are saved in
[results/REPORT.md](results/REPORT.md) and
[results/FINDINGS.md](results/FINDINGS.md), with compact machine-readable
evidence alongside them. Raw traces remain in the artifact directory above.
The serving tree remains the recommended default; neither new checkpoint
established a consistent live throughput improvement.

Regenerate summaries with `analyze.py` and `common_predictions.py` using the
CPU evaluation environment. Then run `.venv/bin/python
experiments/live_predictors_20261008/finalize.py` to audit every request's
scheduled tokens, matched policies, and frozen checkpoints, and copy the
final evidence into this branch.
