# Prediction accuracy versus GPU gaps: retrospective attribution

## Finding

The existing traces do not establish that improved prediction accuracy caused
the large 60K throughput improvement. They rule out avoiding late
prediction-deadline seals as the direct explanation of most of that improvement.
An indirect effect through subsequent batching, transfer submission, or client
arrivals remains possible and requires an intervention experiment.

Scope: actual GLM 5.3 TP8, concurrency 32, one output token; 36 main cells plus
nine conditioned cells, 3,375 measured requests. This new analysis runs on CPU
only and joins 243 native forecasts to 583 consecutive compute transitions.

## Where the 60K gaps changed

Main experiment: mean milliseconds per 75-request episode. Each transition
belongs to exactly one category. A direct prediction must target the immediately
preceding compute batch; a deadline exit must explicitly use
`compute_seal_deadline`. Load-only means `has_compute=False` in the successor's
sealing transaction, not that the preceding compute batch contained zero tokens.

<!-- markdownlint-disable MD013 -->
| Gap category | Previous trees | Matched elapsed | Matched offset |
| --- | ---: | ---: | ---: |
| Load-only successor transaction | 1666.10 | 1633.58 | 606.20 |
| Direct prediction, deadline exit | 142.65 | 8.12 | 46.34 |
| Direct prediction, other exit | 97.91 | 119.12 | 135.72 |
| No direct native prediction, other path | 73.87 | 16.63 | 57.67 |
| Total | 1980.52 | 1777.45 | 845.93 |
<!-- markdownlint-enable MD013 -->

The total old-tree to offset gap reduction is 1,134.59 ms/episode. Of this,
1,059.90 ms, or 93.4%, is in the load-only category. That is an accounting
fraction, not a causal attribution percentage. There are six load-only gaps
across four old-tree episodes versus two across four offset episodes.

In the matched elapsed to matched offset comparison, load-only gaps also
dominate the improvement: their reduction is 1,027.39 ms/episode versus
931.52 ms/episode total. Prediction-deadline transition gaps increase from
8.12 to 46.34 ms/episode in this comparison. Thus the matched feature ablation
also does not show a direct late-deadline explanation.

The conditioned experiment has the same qualitative pattern: load-only gaps
are 2,273.29 ms/episode for old trees, 2,230.69 for matched elapsed, and 1,623.51
for matched offset. Conditioning did not eliminate phase variation.

## Prediction error does change the timer, but not necessarily the gap

For each prediction-deadline exit, the audited identity is:

```text
actual seal lead = configured lead - prediction error - timer overshoot
prediction error = predicted duration - measured duration
```

All 31 prediction-deadline exits across both studies sealed before the measured
critical end of their preceding compute batch. None has a positive exposed
pre-seal interval after that end, using the saved span estimate. The main 60K
old-tree deadline exits seal an average 56.31 ms early, versus 12.99 ms early
for offset. Better accuracy places the timer closer to the intended boundary,
but these cases do not show late seals being removed.

For example, old-tree episode r3, step 3012, underpredicts by 147.20 ms,
seals 151.72 ms before the preceding end, and still has a 140.54 ms GPU gap
after it. A late seal cannot explain that gap. Replaying the offset predictor
on the identical saved input would move its deadline about 135.61 ms later.
It does not tell us when the next batch would start or what membership it
would have with that later deadline: the candidate trace stops at the real seal.

Clock accounting separates each gap into before seal snapshot, snapshot to
membership record, membership record to dispatch, and dispatch to GPU start.
These are disjoint intervals, not independent causes. The snapshot timestamp
precedes final membership accounting; the later record is retained explicitly.

## What creates the large load-only gaps

For main old-tree load-only gaps, the next batch's first selected request
finishes KV transfer on average 103.45 ms after the preceding compute end,
the eighth at 597.61 ms, and the last at 1,049.99 ms. The seal snapshot follows
the last completion by just 0.49 ms. For the offset cases the corresponding
times are 135.68, 592.15, 1,130.34, and 1.29 ms.

These waits follow the completion of KV transfers for the selected batch,
rather than a late prediction deadline. The duration of an individual
load-only gap did not shrink; the offset runs encountered fewer such gaps.
Batch sizes of 16 or 17 are common successors, with a final shorter tail.

Old-tree successors already had final membership recorded before any native
prediction in their episode account for 658.40 ms of gap per episode, versus
284.48 ms for offset. The current episode's predictor cannot cause decisions
made before its first forecast. This does not exclude carryover from an earlier
episode or a later predictor effect on the subsequent gaps.

## Controlled experiment needed for causal evidence

1. Keep one frozen predictor, the optimized preparation path, CPU layout,
   transfer policy, readiness exits, and original sparse selector identical.
   Randomly assign episodes to duration offsets -50, 0, and +50 ms. Apply the
   assigned offset at the scheduler's use of the prediction, not in learned
   inputs. This manipulates timer error without changing model architecture
   or evaluation cost. Record both the unmodified and effective prediction.
2. First use repeated transitions with a controlled preceding batch geometry
   and KV-load submission timing. Vary preceding batch sizes and ready counts.
   This isolates whether changing the timer changes sealing, admissions, and
   the next physical execution start. Readiness exits must remain active.
3. Then use balanced, repeated continuous 300-request 60K cells at concurrency
   32 to measure the total effect, including batching and client-arrival
   feedback. Record the initial phase separately instead of treating a common
   conditioning episode as proof that initial state is identical.
4. Report the full chain: effective error, deadline availability and binding,
   seal lead, admissions and batch geometry, dispatch/start timestamps, then
   subsequent load-only gap count and duration. Include per-episode confidence
   intervals. Predictions that never influence an exit or wakeup provide no
   direct evidence for the timer mechanism.
5. Add a shadow control where the random perturbation is recorded but does
   not affect scheduler decisions. A repeatable effect in active perturbation
   with no corresponding shadow effect supports causality. Compare signs
   separately: underprediction and overprediction can have different effects.

These GPU interventions are proposed, not run in this audit. No numerical
throughput improvement is inferred from fixed-trace deadline replay.

## Verification and limits

All 45 cells were accepted in the original measurement audit. This script
verifies all 583 transitions and 243 forecast joins, deadline arithmetic,
timestamp ordering, and exact nonnegative gap decomposition. Saved aggregate
gaps are reproduced. Small overlaps of estimated critical spans are clipped
to zero gap. Poll bounds are retained; they are much smaller than the roughly
one-second load-only gaps. CUDA event spans also include CPU submission stalls
and required output joins; they are not isolated GPU busy-kernel time.

Per-transition evidence: `transitions.jsonl`. Group summaries and limitations:
`attribution.json`, under `/mnt/localnvme/glm53_prediction_gap_attribution_20261008`.
