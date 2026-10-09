# Independent readiness serving result

Two paired continuous 300-request episodes per arm completed and stopped.
Both arms use the same tensor-major cache, physical merging, packet
alignment, allocator preference, frozen predictor and original selector.
The new readiness policy is the only serving option that differs.

| Mode | Req/s | Mean TTFT ms | p95 TTFT ms | Gaps ms |
| --- | ---: | ---: | ---: | ---: |
| Ordinary dependency | 7.845 | 3912.2 | 4683.0 | 1678.4 |
| Independent stream | 7.462 | 4112.6 | 4917.0 | 4128.7 |

The mean same-seed throughput change is -4.85%, with a regression in both
repeats. Compute rounds fall from 43.5 to 36.5 and the compute envelope
falls from 36,244.6 to 35,710.7 ms per episode. Larger inter-compute gaps
offset that improvement. The per-episode exposed forward-to-GPU-start
interval rises from 1,483.9 to 2,442.3 ms. Other exposed scheduling and
worker intervals also increase. These are phase measurements, not proof
of a particular driver lock, bandwidth limit or kernel cause.

All 600 loads per rank used the guarded independent path. On rank zero,
the original compute stream was busy for 263 and 256 of the 300 loads
in each repeat. The experiment did remove the intended dependency;
that dependency removal did not produce a serving benefit here.

Cache restoration, zero GPU prefix hits, one output token, no preemption,
frozen predictor and all-rank optimized preparation checks passed.
Earlier-dispatch zeroing and copy footprints are tracked using completion
events with bounded history and conservative fallback. CPU guard checks
and the serialized GPU dependency/byte checks are archived separately.
No byte-audit instrumentation was enabled for throughput measurements.

Retain this as an opt-in experimental branch, with its flag defaulting
to false. It is not selected for the resident CPU-cache comparisons.
The raw archive is hash-verified and includes the exact launch commit,
plans, predictor, counters, responses and all-rank timing. These serving
checks do not establish whole-model numerical accuracy or significance.
