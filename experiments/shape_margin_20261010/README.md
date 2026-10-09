# Shape-dependent deadline margin

Branch: `experiment/shape-deadline-margin-20261010`.
Parent: `0c14900`; keep its frozen broader calibration and worker configuration.

The broader duration fit achieves 14.25 ms grouped seed MAE and 16.60 ms
leave-geometry-out macro MAE in development. The first correction failed
crossed geometries. Its independent 300-request confirmation is running
and stays unchanged. Development accuracy does not select a serving winner.

The margin audit separates positive A finish-prediction error from B's
worker-entry-to-forward host setup. The latter excludes FIFO waiting but
includes connector and concurrent-work interference. Short-suffix large
A batches have p90 positive error near zero, while some long-suffix B
shapes have setup p90 above 40 ms. A single margin can be unnecessarily
early for some shapes and insufficient for others.

Possible next mechanism: freeze finish-error quantiles by A shape, and
setup quantiles by B shape. B's selected base shapes already exist in
the scheduler. Published candidates have known context and residual
prefill lengths. Evaluate all possible successor count bins, taking the
largest setup allowance; use no final or future membership information.
Add the two allowances and a small scheduler allowance. Keep bounded
margins and the configured fallback outside support. Do not train online.

Quantiles are empirical allowances, not probabilistic guarantees. A and
B bins cannot be equated. Sparse bins need pooled fallback. Real serving
must establish whether the additional opportunity is useful and whether
late seals or GPU gaps increase. Preserve all base acknowledgments,
readiness/cancellation snapshots, safety cap, and ordered retirement.

CPU margin auditing is permitted during serving. Runtime mechanism
tests and new GPU allocations wait for the current allocation to stop.
