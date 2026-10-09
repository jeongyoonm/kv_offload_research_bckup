# Recalibrate setup lead after native store submission

The native full-submission path reduced exposed pre-forward intervals
from 108.3 to 10.8 ms per 75-request episode. Unclipped worker entry to
forward submission averages 22.0 ms, with a 19.5 ms median and 65.6 ms
maximum across measured batches. Under the unchanged 100 ms setup lead it
recovered fewer batching opportunities and used 12.75 rather than 12.25
compute rounds. The clipped exposed times are not total setup latency.

This comparison keeps the frozen predictor and 90 ms prediction subtraction.
Both arms use the verified native D2H implementation with full submissions,
allocator avoidance and original cached preparation. Only setup lead changes
from 100 ms to 25 ms (or 50 ms via the CLI). The smaller lead is grounded in
observed setup cost, while prediction variance could still make it too late.
Measure actual GPU gaps, admission recovery and compute rounds to decide.

Use two paired continuous 300-request episodes per arm, with 75 CPU prefixes
and four distinct suffixes. CPU restoration on every request and zero GPU
prefix hits are required. Priming is excluded and each episode has one reset.
This is an optional follow-up after the required cache-reference comparisons;
do not start it without enough time to complete and drain within the budget.
