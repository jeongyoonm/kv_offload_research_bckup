# GLM 5.3 TP8 store-overlap exploration

All measurements preserve original model execution, FP8 KV and fixed
cache capacity. Warm priming and startup are excluded. Experiments are
descriptive; the small repeat counts do not establish statistical
significance. Separate controls and acceptance checks are retained.

## Paired 75-request experiments

Each arm uses four independently primed 75-request episodes. The gain
is the mean of same-seed throughput changes. Gaps are conservative
all-rank inter-compute gaps per episode, not an additive CPU breakdown.

| Proposal | Control req/s | Candidate req/s | Gain % | Gaps ms |
| --- | ---: | ---: | ---: | ---: |
| Scoped reuse fences | 7.808 | 7.818 | +0.13 | 145.7 → 217.2 |
| Specific producer events | 7.817 | 7.568 | -3.34 | 144.8 → 528.3 |
| Allocator avoidance | 7.795 | 7.827 | +0.40 | 222.1 → 161.5 |
| Store descriptor sorting | 7.752 | 7.827 | +0.96 | 197.8 → 117.0 |
| Future load reservations | 7.734 | 7.796 | +0.82 | 158.4 → 137.6 |
| Critical stores first | 7.718 | 7.825 | +1.39 | 189.7 → 91.3 |
| Ascending GPU assignment | 7.523 | 7.268 | -3.41 | 575.7 → 1011.0 |
| Preencoded allocator control | 7.802 | 7.784 | -0.23 | 167.8 → 90.6 |

Store sorting merged no descriptors in block-major storage. One
coalescing repeat coincided with unintended GPU test allocations;
its audit is retained and it is not used as evidence for merging.
Preencoded-client results use a changed client harness in both arms
and are interpreted separately from the original arrival pattern.

## Native D2H submission

Modes use independent allocations with the same four seeds and
300 measured requests each. The gain compares matching seed/size
episodes across allocations, rather than alternating in one server.

| Mode | Req/s | Gain % | Gaps ms | Exposed pre-fwd ms |
| --- | ---: | ---: | ---: | ---: |
| Synchronous store control | 7.792 | +0.00 | 152.1 | 108.3 |
| Native store, full submission | 7.873 | +1.04 | 34.0 | 10.8 |
| Native store, 8 MiB chunks | 7.877 | +1.09 | 71.1 | 10.0 |

## Continuous CPU-cache workload

Each allocation uses two continuous 300-request episodes with 75
prefixes and four distinct suffixes per prefix. CPU restoration is
required on every request; GPU prefix hits and preemptions reject
a cell. There is one reset before each episode, with none inside it.

| Mode | Req/s | Gain % | Gaps ms | Mean TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Continuous block-major control | 8.175 | +0.00 | 248.9 | 3759.0 |
| Continuous tensor-major only | 8.175 | +0.00 | 251.2 | 3754.8 |
| Continuous tensor-major plus merging | 7.665 | -6.21 | 3428.6 | 4021.2 |
| Continuous sorted recycled CPU rows | 7.934 | -2.94 | 1296.0 | 3867.0 |

## Guarded independent load readiness

Two matched continuous 300-request episodes per arm: 600 requests
each. Both arms retain the same tensor-major physical optimization.

| Mode | Req/s | Gaps ms | Mean TTFT ms | p95 TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Ordinary dependency | 7.845 | 1678.4 | 3912.2 | 4683.0 |
| Independent stream | 7.462 | 4128.7 | 4112.6 | 4917.0 |

Mean paired throughput change: -4.85%.

## Native store deadline lead

Two matched continuous 300-request episodes per arm. Native full
store submission and the frozen predictor are fixed; only the
setup lead differs. The 90 ms prediction subtraction is fixed.

| Mode | Req/s | Gaps ms | Mean TTFT ms | p95 TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Lead 100 ms | 8.178 | 255.4 | 3753.9 | 4694.3 |
| Lead 25 ms | 8.138 | 292.7 | 3769.4 | 4594.1 |

Mean paired throughput change: -0.49%.

## Resident-cache references

### sg1

| Prompt | Working set | Req/s | Mean TTFT ms | p95 TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| familiar60k | 32 | 7.674 | 3076.8 | 4150.3 |
| short16k | 75 | 25.797 | 1091.6 | 1285.0 |
| familiar60k | 75 | 0.494 | 51335.2 | 64645.8 |

### sg2

| Prompt | Working set | Req/s | Mean TTFT ms | p95 TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| familiar60k | 32 | 7.296 | 3002.2 | 4382.5 |
| short16k | 75 | 25.151 | 1088.8 | 1428.4 |

### sg3

| Prompt | Working set | Req/s | Mean TTFT ms | p95 TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| familiar60k | 32 | 7.341 | 2926.6 | 4403.4 |
| short16k | 75 | 26.046 | 1052.7 | 1376.2 |

### sg4

| Prompt | Working set | Req/s | Mean TTFT ms | p95 TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| familiar60k | 32 | 7.720 | 3066.1 | 4120.7 |
| short16k | 75 | 25.770 | 1090.9 | 1255.8 |

## Background stores on matched CPU cohorts

Only native D2H submission differs between these CPU runs.
Values pool identical request counts and seeds across allocations.

| Prompt | Sync/async req/s | Rate % | Mean TTFT % | p95 % |
| --- | ---: | ---: | ---: | ---: |
| familiar60k | 7.296/7.341 | +0.62 | +2.52 | -0.48 |
| short16k | 25.151/26.046 | +3.56 | +3.31 | +3.65 |

## Matched resident working sets

Independent allocations use identical seeds, request counts,
precision and cache capacities. GPU-only sg4 verifies full
prefix hits and process-local THP disabling on every worker.
CPU runs reset GPU cache and require full CPU restoration on
every request. Mean and p95 TTFT pool measured requests.

| CPU run | Prompt | CPU/GPU req/s | Rate % | Mean TTFT % |
| --- | --- | ---: | ---: | ---: |
| sg2 | familiar60k | 7.296/7.720 | -5.49 | +2.08 |
| sg2 | short16k | 25.151/25.770 | -2.40 | +0.20 |
| sg3 | familiar60k | 7.341/7.720 | -4.91 | +4.55 |
| sg3 | short16k | 26.046/25.770 | +1.07 | +3.50 |

GPU resident references have ordinary scheduling and preparation;
they are measured references, not mathematical upper bounds. The
75-prefix 60K GPU-only case exceeds the unchanged GPU capacity.
The initial sg1 run did not explicitly audit process-local THP;
the corrected sg4 reference sets and checks it on all GPU workers.

## Progress and provenance

- sf1: complete — Scoped reuse fences
- se2: complete — Specific producer events
- sa1: complete — Allocator avoidance
- sc1: complete — Store descriptor sorting
- sr1: complete — Future load reservations
- sp1: complete — Critical stores first
- so1: complete — Ascending GPU assignment
- sn0: complete — Synchronous store control
- snf: complete — Native store, full submission
- sn8: complete — Native store, 8 MiB chunks
- st0: complete — Continuous block-major control
- st1: complete — Continuous tensor-major only
- stm: complete — Continuous tensor-major plus merging
- stc: complete — Continuous sorted recycled CPU rows
- si1: complete — Independent load ready stream
- sv1: complete — Preencoded allocator control
- sdo1: not_started — Preserved descriptor order
- snl: complete — Native D2H setup lead recalibration

Raw inputs, outputs, cache counters, timing events, launch commands,
ownership records and failure logs remain under the short run roots.
Verified archives and results are committed on each proposal branch.
See plan.json for branch/workspace mapping and the process logs for
the chronological sequence. Byte tests validate transfer geometry;
they do not establish model accuracy for every serving workload.
