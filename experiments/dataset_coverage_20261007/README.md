# GLM 5.3 TP8 dataset coverage study

Branch: `experiment/glm53-dataset-coverage-20261007`.
Artifacts: `/mnt/localnvme/glm53_dataset_coverage_20261007`.
Base: `81e46f8`, including the opt-in private CPU KV cache initialization.

Keep GLM 5.3 NVFP4, TP8, FP8 KV, 96 GiB GPU KV per rank, the 2.13 TB
aggregate CPU cache, 64-token blocks, process-local THP disabling, and NUMA
placement. Use the original sparse-attention selector. All new cells use the
same private CPU layout; this is not an initialization-layout throughput study.

Actual per-step context and new prompt tokens define compute geometry. CPU-hit
percentage only generates workload mixtures. The compute-duration target is
`max(rank_start_upper + CUDA_span) - max(rank_start_upper)`, preserving the
historical target. It includes required forward/output completion, excludes the
preceding prefix-load wait, and is not a sum of isolated kernel times.

## Data protocol

`matrix.py` deterministically freezes geometry, composition, transfer, familiar
volume controls, development seeds, and final test seeds before labels exist.
Targets are 400 causal forecasts per expansion block and 1,200 familiar controls.
Geometry/composition cells use a queued cohort. Transfer cells first load all
target prefixes into GPU HBM behind a default-off scheduling gate, start unrelated
loads, and release target compute immediately or after 10 ms. Background requests
never become compute members and are aborted after target completion. The gate
does not modify model execution or insert a wait inside its timed interval.

An independent observer records the first usable all-rank active start snapshot.
It does not invoke the predictor, modify scheduler membership, or retrain. Runtime
features are reconstructed using load submissions and engine completion receipts
at the snapshot timestamp. Future events only supply labels and overlap audits.
Very short batches without a causal snapshot remain base-only records.

Validate actual token accounting, CPU-hit counts, zero GPU-prefix hits, and no
preemptions. Retain every rejected cell. Keep profiler/JIT and diagnostic selectors
out of measured labels. Extend extraction to explicitly classified all-prefill
batches with zero prior context; default legacy extraction remains cached-only.

## Budget and evaluation

Four hours of TP8 allocation, including starts/priming, with an independent
shutdown timer. Reserve 30 minutes for pilot, 120 for collection, 45 for shadow
evaluation, and 45 for live confirmation. CPU fits use historical data plus matched
new base/forecast budgets, fixed tree recipes, three training seeds, and episode
cross-fitting. Component removals replace data with familiar samples to preserve
sample count. No test labels enter selection. Report accuracy by workload group,
directional deadline errors, forecast coverage, paired uncertainty, and historical
regression. Compare selected and historical predictors with identical live policy.

Run each addition both with the original weighting and with a source-mass-matched
control. The latter keeps historical weights independent of new workload counts
and gives every new sample budget the same total weight. Record both the frozen
tree's broad-shape extrapolation and the old cached-only guard's linear fallback.
Only development accuracy selects the live checkpoint. Dataset variants receive
separate branches with frozen checkpoint hashes; they share the same collector,
features, and tree hyperparameters.

The geometry expansion also reuses already-recorded cold prefix-priming batches
from repeated training cells, after excluding first-use and JIT windows. This
extends uncached suffix coverage beyond 8K without another GPU run. These records
share an episode with their associated measured cohort for cross-fitting, and
receive separate cold workload labels for balanced sampling. No development or
final priming labels enter training. The total matched addition budget stays fixed.

## Commands

```bash
.venv/bin/python -m pytest experiments/dataset_coverage_20261007/verify.py -q
.venv/bin/python experiments/dataset_coverage_20261007/run.py
.venv/bin/python experiments/dataset_coverage_20261007/run.py --normalize
.venv/bin/python experiments/dataset_coverage_20261007/finish.py
```

Raw process events, configuration, request seeds, counters, response records, and
timing traces are persisted after each cell. Checkpoints are never retrained at
prediction time. Report incomplete phases explicitly when the hard budget expires.
