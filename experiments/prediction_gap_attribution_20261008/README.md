# Prediction accuracy and GPU gap attribution

CPU audit of the 36 main and nine conditioned GLM 5.3 TP8 cells. It joins
243 native forecasts to 583 consecutive compute transitions. The serving
implementation and frozen checkpoints are unchanged.

Run from this checkout:

```bash
.venv/bin/python experiments/prediction_gap_attribution_20261008/analyze.py \
  --output /mnt/localnvme/glm53_prediction_gap_attribution_20261008
```

`attribution.json` contains per-group summaries. `transitions.jsonl` contains
each predecessor, successor, prediction, seal snapshot, membership record,
dispatch and GPU start. Assertions verify eight-rank completed spans, timestamp
ordering, nonnegative disjoint gap phases, and the exact deadline arithmetic.
Small overlaps between estimated critical spans are clipped to zero gap.

The clock decomposition identifies when time was spent. It cannot identify
an independent causal effect of prediction accuracy on throughput. In
particular, load-only gaps can depend on earlier decisions and client arrivals.
Candidate readiness logging ends at the real seal. Hypothetical deadlines
after that seal have no reported candidate readiness count.

Findings and a controlled intervention proposal are in
`results/FINDINGS.md`. Full evidence is under
`/mnt/localnvme/glm53_prediction_gap_attribution_20261008`.
