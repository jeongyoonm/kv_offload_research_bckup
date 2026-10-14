# Productive load-only early exit

Branch: `experiment/shape-productive-load-only-20261010`.
Parent: `f0ccfea`; retain frozen duration calibration and selected workers.

The first 300-request short-context shape-calibration cell improved MAE
58.7 to 7.3 ms and reduced compute rounds 39 to 30, yet slowed serving
28.47 to 26.70 req/s. Its 8/12/12 cycle exposed 1.72 s of inter-compute
gaps, versus 0.55 s for control. Largest gaps wait for eight of sixteen
published loads with no compute deadline. A same-shape guarded cell
produced a staggered sequence and only 39 ms of gaps, showing strong
cadence sensitivity. All cells and reversed repeats remain required.

Prepare an opt-in early exit only when no compute is queued. Estimate
batch duration using the frozen direct attention-work ridge fit. Its
positive suffix and prefix-index terms define normalized uncached work.
Compare the ready subset's work rate to the full already-published
cohort's work rate. Sweep 70% and 80% empirical productivity thresholds.
This is a model-based allowance, not measured GPU duty cycle or an
optimality guarantee. It uses base, actual ready and published shapes;
never future readiness, membership, or labels. Mixed suffix/context
work is weighted rather than treating every request as equal work.

The existing half-cohort exit remains: this policy only seals earlier.
Require at least one all-rank-acknowledged KV-ready candidate and all
base acknowledgments. Disable the new exit after cancellation; retain
the normal cancellation policy. Unknown/unsupported shapes retain the
old behavior. Preserve safety timeout, snapshot finality, allocation,
connector accounting, FIFO execution and ordered retirement. Schema 5
adds this policy without a shape margin; schema 6 can combine them in
a later explicit ablation. Neither is enabled in the running `yc3`.

Run the inherited and new CPU mechanism checks after `yc3` stops,
before any serving cell loads the new artifact. Fresh matched 300
cells must test batching, exposed waits, KV readiness during compute,
prediction error, throughput and TTFT. Retain unsupported or regressing
cases; do not select by admission ratio or compute rounds alone.
