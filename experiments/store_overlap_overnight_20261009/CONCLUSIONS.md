# Store blocking: conclusions and boundaries

The selected design combines allocation avoidance with native background
D2H submission. It preserves the latest scheduler, input preparation,
chunked H2D, model, weights, sparse selector and FP8 KV configuration.
Both additions remain experimental and disabled by default.

## Why stores blocked forward

The free pool selected recently freed GPU rows while stores still read them.
This happened with thousands of other free rows available. Choosing rows
without pending readers removed the observed reuse dependencies: 7,536
pending-source allocations became zero, as did 224 all-rank reuse fences.
Under memory pressure, the implementation falls back to a real completion
wait. It does not discard the dependency.

Separately, store submission called CUDA on the main forward thread. With
4,752 descriptors per typical suffix store, the API consumed roughly 3.2 ms
per job. A disjoint-buffer probe showed additional API time during concurrent
H2D submission. This locates the cost inside the API but does not identify a
specific internal driver lock. Moving submission to a native worker reduced
main-thread enqueue time to roughly 0.02 ms per job.

## What improved

Four matched-seed, independently allocated 75-request episodes per mode gave
the native full-submission mode a descriptive 1.04% throughput improvement:
7.792 to 7.873 requests/s. Conservative inter-compute gaps fell from 152.1
to 34.0 ms per episode. Exposed pre-forward intervals fell from 108.3 to
10.8 ms; these are clipped intervals within gaps, not total setup cost.
Unclipped worker-entry-to-forward setup averaged 43.7 versus 22.0 ms per
batch. Residual gaps and setup remain.

The small throughput change is consistent with most original store API work
already overlapping compute. The unchanged scheduler also produced slightly
more compute rounds and lower opportunity recovery after the worker became
faster. Removing a host cost does not imply an equal serving speedup.

Allocator avoidance proved its mechanism in a second client-controlled run,
but its small throughput gain did not reproduce: the change was -0.23%.
It is retained to avoid unnecessary dependencies, without a standalone
throughput promise. Native full and 8 MiB store modes are not statistically
distinguished by these four repeats.

## Why other proposals were not selected

Tensor-major CPU storage enabled real descriptor merging and improved a
transfer microbenchmark. Layout alone was neutral in continuous serving.
Merged serving had one 12.9% regression with larger load-only gaps despite
substantial descriptor reduction. Sorting recycled CPU rows did not recover
the loss. Guarded independent load readiness removed the intended dependency
but regressed both matched continuous repeats, averaging -4.85%. Narrower
producer events and GPU assignment sorting also lacked reliable gains.
These results remain available on their separate branches.

## Safety and measurement scope

Actual producer completion orders each store. Private descriptors and buffers
remain owned until completion; reset and shutdown drain real transfers.
Exactly-once retirement and byte-level transfer checks passed in targeted
CPU/GPU tests. Serving checks validate requests, prefix restoration, lack of
preemption, frozen predictor state and preparation acknowledgments. They do
not establish whole-model accuracy or remove selector nondeterminism.

GPU-only runs measure a resident reference under the same GPU cache budget.
The full 75-prefix 60K working set exceeds that budget; its capacity-limited
result is separate from the smaller all-hit comparisons. GPU-only removes
connector-dependent scheduling and uses ordinary preparation, so its result
is a measured reference rather than a mathematical upper bound.

See `REPORT.md` and its JSON companion in the data directory for all tables,
including matched resident controls and any deadline-lead experiment that
completed inside the overnight allocation. See `plan.json` for branches,
timed source commits and verification provenance.

## Final follow-up

Reducing setup lead from 100 to 25 ms improved admission recovery from
95.28% to 98.65% and consolidated one compute round on average, but mean
paired throughput regressed 0.49% and GPU gaps increased. Keep 100 ms for
the selected worker. This two-repeat study does not identify a universal
best lead. See `DEADLINE_LEAD_FINDINGS.md` in the data directory.

All 21 completed allocations passed launch/cache acceptance, covering
13,280 measured requests. One earlier sorting repeat overlapped GPU tests;
its quality exception and clean three-pair result remain explicit. All
owned servers stopped and the final GPU process query was empty.
