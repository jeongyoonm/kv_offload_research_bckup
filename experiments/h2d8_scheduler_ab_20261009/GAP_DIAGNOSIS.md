# Why the 60K gaps increased

This audit uses the existing four paired 75-request episodes. It launches
no GPU jobs and changes no serving code.

## Measured phase attribution

Mean exposed time per episode, in milliseconds. Each phase is clipped to
the interval between A's conservative all-rank execution end and B's
conservative all-rank execution start. The phases sum to the previously
reported inter-compute gaps.

| Phase | C | D |
| --- | ---: | ---: |
| Before dispatch, including scheduler work | 31.0 | 72.4 |
| Dispatch to worker entry | 12.4 | 7.3 |
| Request state, inputs, and attention metadata | 8.4 | 27.3 |
| Metadata ready to forward submission | 36.8 | 272.7 |
| Forward submission to observed GPU start | 8.6 | 4.8 |
| Total | 97.1 | 384.5 |

The metadata-to-forward phase accounts for 236.0 ms, approximately 82%
of the additional 287.4 ms gap. Token packing and RPC delivery are not
the main source of the increase.

The source path in this phase handles KV connector pre-forward work.
`ActiveKVConnector.pre_forward` calls `handle_preemptions`, which submits
deferred GPU-to-CPU stores and waits for `jobs_to_flush` when needed.
It then starts new loads. The asynchronous submission worker and chunk
configuration used in this experiment apply to H2D; stores retain the
ordinary worker submission path.

The largest stalls include two batches with zero new H2D load jobs:
D repeat 1 step 1204 and repeat 2 step 1908. Their all-rank exposed
metadata-to-forward times are 101.2 and 109.1 ms. This rules out new
H2D load submission as the explanation for those stalls. It points to
deferred store handling or its completion fences. The existing markers
do not separate store descriptor construction, CUDA API submission,
and block-reuse flush waits, so exact attribution within the connector
still needs finer instrumentation. Other short operations also share
this bracket; it is not a direct per-call connector timer.

## Lost overlap and inaccurate deadlines

In C, B's forward submission occurred before A's measured end in 49/56
compute transitions, or 87.5%. In D it did so in 12/44, or 27.3%.
Holding membership delays the authoritative execute RPC and therefore
delays the ordinary worker's connector handling. Work which can overlap
A becomes exposed if B is dispatched near or after A finishes.

The compute-deadline audit finds 42 recorded deadlines for the 44 D
transitions. Of those, 39 fall after A's measured end. The mean deadline
error is +83.7 ms and median +94.3 ms, already including the 5 ms lead.
Thus the implied finish prediction averages about 88.7 ms late.

Twenty-four transitions seal on `predecessor_compute_finished`, and
twenty on `candidate_target_ready`. None of these 44 transitions seals
on the compute deadline. The predictor deadline therefore does not
provide the intended advance on these 60K transitions.

The checkpoint was frozen for the worker ablation. Retargeting its
chunk-size compatibility identity did not recalibrate the model. These
records demonstrate live overprediction under the 8 MiB configuration;
they do not isolate the chunk-size change as the sole source of that
prediction error. Batch composition also differs between C and D.

## Implications

The measured mechanism is late dispatch exposing connector work. This
does not establish a new preparation-thread lock contention problem.
Two improvements address it:

1. Calibrate predictions on the new configuration and budget the entire
   remaining worker path in the seal lead. A 5 ms lead does not cover
   observed connector stalls around 70-110 ms.
2. Submit deferred stores independently of B's final membership, using
   the existing CUDA ordering and block-reuse completion fences. This
   could retain admission time while moving submission off B's critical
   path. Its benefit requires measurement; fences cannot be removed.

The first implementation step should instrument deferred store submission
and store-flush waiting separately. It will distinguish CPU submission
overhead from a necessary wait before reusing GPU blocks.

## Reproduction and evidence

```bash
PYTHONPATH="$PWD" .venv/bin/python \
  -m experiments.h2d8_scheduler_ab_20261009.gap_audit \
  /mnt/localnvme/h2d8_scheduler_ab_20261009/paired_v2
```

`gap_audit.json` retains the scheduler boundaries, selected worker-stage
timestamps for every rank, new-load submission timestamps, and deadline
offsets for each transition. The audit checks complete stage coverage,
trace overflow, chronological boundaries, predecessor identity, and
exact agreement of the phase sum with each measured gap.
