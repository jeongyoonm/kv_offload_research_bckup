# Findings: latest offset predictor deadline comparison

## Completed experiment

GLM 5.3 NVFP4, TP8, concurrency 32, one output token. The two arms use
identical frozen `matched_offset` predictor states and identical optimized
preparation, earlier KV submission, chunked async H2D, and readiness policies.
Only deadline enforcement versus shadow recording differs, with a 5 ms lead.
No serving implementation, model weights, architecture, or selector changed.

Eight paired 75-request episodes per arm for 60K; four pairs per arm for 16K
and mixed cached/cold traffic. Total: 32 cells, 2,400 measured requests.
Six 32-request warmups and every reset/priming interval were excluded.
These episodes are not uninterrupted 300-request streams and do not reproduce
historical run conditions closely enough to rank against earlier 7.1 req/s cells.

## Throughput and TTFT

See `REPORT.md` and `comparison.json` for complete tables and paired intervals.
The 60K aggregate gain was 6.28%:
6.371 to 6.771 req/s. Its paired 95% interval
was [-0.09%, +13.05%].
The 16K gain was 4.75%, while mixed traffic was essentially unchanged.
All three intervals include zero. The average gains are encouraging, but this
allocation does not establish a statistically clear throughput improvement.
Intervals are unadjusted paired log-ratio Student-t intervals, with episodes as
the sampling unit; one allocation cannot capture between-allocation variance.

For 60K, mean TTFT fell 5.77%:
4317.5 to 4068.5 ms. P95 was effectively unchanged:
5474.5 versus 5474.9 ms. One-token output workloads
measure prefix restoration and prefill/first-token service, not long decoding.

## Mechanism evidence and limits

Mean 60K inter-compute gap fell from 1991.8 to
1004.2 ms per episode, a
49.6% reduction.
The lower/upper timing bounds are narrow (see the JSON aggregates).
Most of the reduction is in the load-only category:
1780.9 to
719.7 ms per episode.
Deadline-associated gaps added 84.2 ms
per episode. Categories are an accounting of trace outcomes, not a proof that
individual timer firings caused all later reductions. Timing spans include CPU
submission/output joins and do not measure isolated kernel busy time.

Compute rounds increased from 80 to 86, while
late admissions fell slightly from 583 to 579.
The average requests per compute round therefore changed from
7.50 to 6.98.
The gain does not come from consolidating requests into fewer compute rounds.

There were 12 explicit deadline exits in 60K,
4 in 16K, and none in mixed traffic. Mixed had only four issued forecasts per
arm, so it exercised very few prediction/deadline decisions.

`batch_phase_audit.json` records every compute batch and those sealed before the
first forecast. Several slow episodes in either arm began with a 17-request
batch before prediction could act. Initial formation varies despite matched
seeds, and this variation contributes to uncertainty. The alternating-order,
matched-work experiment estimates the policy's average effect; it cannot assign
every individual episode difference to a hard deadline event.

The recorded KV-ready/recovered counts overlap any measured compute span,
then intersect that request set with all late admissions. Their denominator is
policy-dependent and is not the immediate predecessor of each transaction.
Do not interpret those fields as a per-transaction predecessor admission ratio.

## Predictor and accounting audit

All 184 native forecasts replayed exactly from saved issuance features and
execution snapshots. Both arms used the same checkpoint, excluded elapsed time
from the learned model, and subtracted timing only in deadline arithmetic.
For 60K, issued MAE was 14.52 ms off and
14.74 ms on; these policy-dependent populations are
not an architecture comparison. No retraining occurred.

All 32 cells passed CPU-restoration counts, zero GPU-prefix-hit/preemption
checks, all-rank preparation coverage, exact per-request uncached-token
accounting, unique ordered completion, and one-output-token checks. H2D bytes
were equal within every pair. Measured intervals contained no compilation or
runtime errors. These checks do not establish semantic output equivalence;
the authorized original sparse selector remains nondeterministic.
The CPU deadline tests passed: 15 tests, 175 deselected.

## Reproduction and storage

Branch: `experiment/glm53-offset-deadline-ab-20261008`.
Checkout: `/mnt/localnvme/vllm_glm53_offset_deadline_ab_20261008`.
Artifacts: `/mnt/localnvme/glm53_offset_deadline_ab_20261008`.
The preserved rejected startup preceded measurements and was repaired by
restoring native dependency links; see `workspace_setup.json`.
The driver and server commands, process log, per-cell results/traces, checkpoint,
source provenance, and final ownership cleanup are retained in the artifact root.
