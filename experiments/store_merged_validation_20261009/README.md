# Continuous serving validation

Run this only after the individual physical-layout proposals have measured
benefits. Both arms retain the latest scheduler, preparation, global GPU reuse
fences, cache capacities, original selector, model and KV precision. The
candidate enables store-aware allocation, ascending selected GPU assignment,
H2D/D2H descriptor merging and packet-aligned splitting together.

Each episode primes 75 distinct 58,368-token CPU prefixes once. It then runs
300 closed-loop requests at concurrency 32, cycling through those prefixes
with four distinct 3,072-token suffixes per prefix. There is one GPU reset before
measurement and no reset or re-prime during the 300 requests. The logical cache
demand, including rounded generated-token and prime tails, remains below the
87,220-block CPU capacity. The GPU pool cannot retain the 75-prefix working set;
every request must still report the expected CPU restoration and GPU hit counters
must remain unchanged, or the cell is rejected.

This checks whether descriptor/submission benefits survive continuous operation.
Two alternating repeats per arm are descriptive. Post-response trace-drain time
includes the draining policy RPC and trace file reads, and is reported separately
from request throughput. The existing 75-request mode is retained for a matched
combined comparison if the continuous working set fails validation.

Implementation and byte tests are inherited from the allocator, physical-order
and packet-alignment branches. This branch changes the workload harness and
policy composition only; it introduces no additional model execution change.
