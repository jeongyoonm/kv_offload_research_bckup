# Allocation order serving findings

All eight cells completed with the same capacity, original model execution,
exact expected CPU prefix restoration and no preemptions. Each arm measured
300 requests over four independently primed 75-request episodes.

Mean paired throughput change was -3.41%, with individual changes -12.04%,
-12.07%, +0.87%, and +11.58%. This noisy experiment does not establish a
benefit or a consistent causal penalty from ascending GPU assignment.

Mean inter-compute gaps increased from 575.7 to 1011.0 ms; the compute
execution envelope remained approximately 9.02 versus 9.01 seconds.
Large load-only gaps occurred in both arms. Both arms avoided allocations
from pending stores and retained approximately 5 ms forward-to-GPU delay.

No physical descriptors merged: each prefix load retained 90,288 descriptors
and each suffix store retained 4,752. The CPU private layout strides across
all layers between rows of one tensor, so sorting GPU block assignments
alone cannot create contiguous source and destination pairs. Coalescing was
enabled, but this experiment did not test effective descriptor reduction.

The full raw trace archive, replay inputs, conservative gap analysis and
member hashes are retained in `artifacts`. Trace timing and request checks
do not constitute a model accuracy guarantee or a statistical confidence
interval. Separate tensor-major mapping and descriptor-order proposals
address the measured layout limitation.
