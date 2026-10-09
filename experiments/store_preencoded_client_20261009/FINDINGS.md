# Client-controlled allocator retest

Both arms preencode client JSON outside the measured interval and use an
initial 32-request barrier. The server configuration and frozen predictor
are fixed. Four alternating, independently primed 75-request episodes per
arm give 300 measured requests each. All acceptance checks passed.

| Mode | Req/s | Mean TTFT ms | p95 TTFT ms | Gaps ms |
| --- | ---: | ---: | ---: | ---: |
| Original allocation | 7.802 | 3544.6 | 4519.4 | 167.8 |
| Avoid pending stores | 7.784 | 3572.3 | 4841.7 | 90.6 |

Mean paired throughput change is -0.23%, with mixed signs across repeats.
The trace confirms pending-store allocation instances fall from 7,632 to
zero. Exposed forward-to-GPU-start intervals fall from 66.9 to 5.2 ms per
episode. Compute rounds average 12.0 and 11.75. Removing the reuse wait
is a demonstrated mechanism improvement, but its small throughput gain
in the earlier harness is not reproduced here.

Interpret this separately from earlier absolute rates: the arrival
harness differs in both arms. Gaps and phases use conservative all-rank
bounds. Tail values are means of episode p95 values. Four repeats do not
establish significance, equivalence or a universal throughput effect.
The opt-in allocator preference remains appropriate to avoid unnecessary
reuse dependencies when free alternatives exist, with the safe waiting
fallback retained when source reuse cannot be avoided.

The hash-verified archive retains raw traces, exact source provenance,
plans, counters, predictor and responses. Original model execution, FP8
KV and cache capacities are unchanged. Cache/accounting checks are not
a whole-model numerical accuracy claim.
