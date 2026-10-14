# Fixed prediction correction and B setup reserve

## Experiment

Completed all 16 measured GLM 5.3 TP8 episodes: four matched 75-request
episodes per arm, 300 measured requests per arm. Each uses 61,440 prompt
tokens, a 58,368-token CPU prefix (95%), a 3,072-token uncached suffix,
concurrency 32, and one output token. Four separate 32-request warmups
are excluded. These are reset/primed episodes, not continuous 300-request
runs. All arms retain the improved 8 MiB/depth 2 async H2D worker, early
submission, optimized provisional preparation, readiness exits, 50%
load-only target, efficient waiting, and a frozen predictor.

Constants were chosen from the preceding allocation before this study:
subtract 90 ms from A's prediction and/or replace the 5 ms deadline lead
with a fixed 100 ms estimate of B's setup. This is not a dynamic setup
estimator or a retrained predictor. Membership, accounting, CUDA stream
ordering, and block-reuse fences are unchanged.

Source at server launch: `5827ae5e476cd41f92945fdad51667df05c1ad5e`.
Branch: `experiment/h2d8-deadline-bias-setup-20261009`.
Code checkout: `/mnt/localnvme/vllm_h2d_interference_20261009` (symlink to
`/mnt/localnvme/jeongyoon/vllm_h2d_interference_20261009`).
External artifacts: `/mnt/localnvme/h2d8_deadline_bias_20261009/paired`.

## Results

Levels are arithmetic means across episodes; percentage changes are
geometric means of matched per-repeat ratios versus control. Four
descriptive repeats do not establish statistical significance.

Arm settings are A subtraction/B setup lead: control 0/5 ms, bias 90/5 ms,
setup 0/100 ms, and both 90/100 ms. TTFT is the episode mean; gaps are
inter-compute milliseconds per episode.

| Arm | req/s | Change | TTFT ms | p95 ms | Gap ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control | 7.566 | — | 3617.4 | 4752.3 | 422.9 |
| Bias only | 7.530 | -0.47% | 3610.5 | 4634.3 | 424.4 |
| Setup only | 7.635 | +0.91% | 3587.4 | 4554.6 | 371.3 |
| Both | 7.781 | +2.86% | 3549.2 | 4746.9 | 222.3 |

Combined per-repeat throughput improvements are +4.44%, +4.48%, +1.50%,
and +1.05%. Setup-only changes are +3.41%, +2.47%, +0.28%, and -2.41%.
Bias-only changes are +0.81%, +0.44%, -2.99%, and -0.11%.

Combined mean TTFT improves 1.89%. Its paired p95 TTFT change is +0.12%,
essentially unchanged descriptively; the arithmetic mean of the episode
p95s is slightly lower. Do not present the combined arm as a tail-TTFT
improvement. Setup-only paired p95 TTFT improves 3.92%.

## Prediction and timing mechanism

In control, 35/37 recorded compute deadlines are after A's measured end.
Their mean offset is +83.8 ms, including the existing 5 ms lead. Mean
finish-prediction error is +88.8 ms, independently reproducing the prior
approximately +89 ms finding. No control compute transition seals on
the compute deadline.

Bias alone removes the mean finish error (+1.6 ms), but absolute error
remains 40.3 ms and the 5 ms setup budget remains too small. Its total
inter-compute gap is effectively unchanged, and throughput does not
improve. Correcting mean bias does not remove prediction variance.

With both adjustments, mean corrected finish error is -0.1 ms, MAE is
32.0 ms, and all 41 recorded compute deadlines precede A's measured end
(mean lead 100.1 ms). Twenty-four compute transitions seal on deadline,
versus zero in control. Conservative recorded seal timestamps after A's
end fall from 18/44 transitions to 1/45. This includes ordinary/fallback
transport records and is distinct from deadline error.

B is dispatched before A ends in 20/44 control transitions (45.5%) and
43/45 combined transitions (95.6%). The combined arm therefore actually
advances the ordinary worker path. Its observed inter-compute gaps fall
47.4%, a 200.6 ms reduction per episode.

The newly added direct call markers distinguish deferred store submission
from explicit store-flush waiting:

| Mean diagnostic (ms/episode) | Control | Both |
| --- | ---: | ---: |
| Full deferred store submission | 275.6 | 260.0 |
| Exposed deferred store submission | 191.7 | 31.8 |
| Exposed explicit store-flush wait | 22.0 | 93.7 |
| Exposed complete pre-forward phase | 275.8 | 179.7 |
| Exposed scheduler phase | 80.4 | 12.1 |

Full store-submission wall time changes much less than its exposed part:
the adjustment mainly moves submission behind A. Exposed submission
drops 83.4%. Explicit flush waiting increases, showing that earlier
dispatch cannot remove every block-reuse dependency. Pre-forward falls
96.2 ms; the remaining gap reduction comes from scheduler and other
worker phases. No fence was removed or relaxed.

Store diagnostics sum the largest per-rank duration for each compute
transition. Their rank maxima can differ, so they are not an additive
partition of the all-rank pre-forward phase. A `submit_store` timer
includes descriptor work and CUDA API time, including any internal
blocking; it is not pure CPU active time. The host-phase partition does
sum exactly to each measured inter-compute gap. Overlapping conservative
execution envelopes count as zero gap and are recorded separately.

## Batching tradeoff

| Mean or aggregate ratio | Control | Bias | Setup | Both |
| --- | ---: | ---: | ---: | ---: |
| Compute rounds per 75-request episode | 12.00 | 12.00 | 12.00 | 12.25 |
| Late-admitted requests per episode | 71.50 | 72.25 | 72.25 | 68.00 |
| KV-ready requests during compute per episode | 70.25 | 70.00 | 71.00 | 72.75 |
| Recovered / compute-period KV-ready | 99.64% | 100% | 99.30% | 91.41% |

Combined throughput improves despite slightly more rounds and fewer late
admissions. The gain here comes with reduced gaps, rather than additional
batch consolidation. The opportunity ratio intersects requests admitted
at any seal with requests whose all-rank KV completion occurred inside
any measured compute interval. It is not a per-seal optimality bound.
The policy changes the compute intervals and therefore the denominator.

## Conclusion and limits

The experiment supports budgeting both a corrected A finish and B's
remaining setup. A more accurate mean prediction alone was insufficient
with a 5 ms lead. A fixed 100 ms setup estimate alone was inconsistent.
Their combination improved throughput in all four matched repeats and
directly reduced exposed store submission.

This does not establish a universal 90/100 ms configuration. The predictor
is frozen, variance remains, and earlier sealing loses some opportunities.
A calibrated predictor for the actual worker configuration and an estimate
of B's remaining work remain more general designs to evaluate. There is
no fresh worker-only C arm in this allocation, so it does not establish
closure of the earlier C/D gap by a contemporaneous matched comparison.

## Verification and reproduction

- CPU seal harness: 206 passed, 8 skipped.
- Selected readiness/configuration checks: 15 passed.
- Connector store/flush checks: 8 passed; tracing preserves exactly-once
  submission before the flush fence, with markers enabled and disabled.
- All 16 measured and four warmup episodes passed expected CPU restoration,
  zero unexpected GPU prefix hits/preemptions, one output token, frozen
  predictor state, and preparation coverage on all eight ranks.
- Normal pre-commit hooks passed, including mypy. No model architecture,
  weights, or sparse selector changes. These serving checks do not establish
  semantic model accuracy or replace a byte-level DMA audit.
- Startup spent approximately 22 minutes rebuilding FlashInfer kernels
  after workspace relocation. Startup, priming, and warmups are excluded.
  Stack sampling occurred only during startup, not measurements.
- The API, engine, and all eight owned workers have exited. All GPUs show
  zero allocated memory and utilization, with no compute processes. Shutdown
  emitted inherited resource-tracker cleanup warnings; final cleanup was
  independently checked and recorded in `results/cleanup.json`.

Generate the report from the external artifacts, or extract the committed
raw archive to a fresh directory and pass that directory instead:

```bash
PYTHONPATH="$PWD" .venv/bin/python \
  -m experiments.h2d8_deadline_bias_20261009.report \
  /mnt/localnvme/h2d8_deadline_bias_20261009/paired
```
