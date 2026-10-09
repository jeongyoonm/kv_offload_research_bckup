# Native D2H serving findings

All three independent allocations completed and stopped. Every mode measured
four same-seed primed/reset episodes of 75 requests, with the exact expected
CPU prefix restoration, zero GPU prefix hits, one output token and no preemption.
All modes retain allocator avoidance and global GPU reuse fencing.

| Mode | Req/s | Gain % | Gaps ms | Pre-forward ms |
| --- | ---: | ---: | ---: | ---: |
| Synchronous | 7.792 | 0.00 | 152.1 | 108.3 |
| Native full | 7.873 | +1.04 | 34.0 | 10.8 |
| Native 8 MiB | 7.877 | +1.09 | 71.1 | 10.0 |

Gains are mean same-seed throughput changes across separate allocations.
Gaps and host boundaries are conservative all-rank diagnostics per episode,
not additive latency components. Four descriptive repeats do not establish
statistical significance or equivalence between the two native modes.

Native submission removed most main-worker store API work: summed conservative
store enqueue envelopes fell from 272.4 ms to about 1.4 ms per episode. Rank-0
per-job enqueue means in the first two full-submission repeats fell from about
3.2 ms to 0.02 ms. Most original API work already overlapped computation, so
its removal did not translate directly into throughput. The full mode's mean
compute envelope increased from 9141.6 to 9203.1 ms and compute rounds increased
from 12.25 to 12.75. The 8 MiB mode used 12.5 rounds and a 9157.2 ms envelope.

Mean TTFT was 3542.7 ms synchronous, 3505.9 ms full and 3500.4 ms chunked.
Mean episode p95 TTFT was 4860.3, 4732.6 and 4675.1 ms respectively. Recovery
of KV-ready-during-compute opportunities was 92.83%, 91.50% and 90.44%.
Both native modes improved serving, while batching became slightly less
consolidated under the unchanged predictor and deadline configuration.

The implementation is opt-in: `async_d2h_submission` defaults to false.
Source access remains STREAM ordered after the actual recorded producer.
Descriptors and buffers are private until native completion. Pending source
reuse conservatively waits for actual completion on the host; an unrecorded
future CUDA event cannot safely act as a GPU fence. Allocator avoidance keeps
this fallback out of these capacity-sufficient cells. Reset and shutdown drain
real copies, with FIFO and exactly-once retirement.

CPU regression and targeted lifetime tests passed; three real GPU cases passed
before serving, including producer ordering, FIFO and shutdown/drain behavior.
Raw serving acceptance checks are not an additional full model accuracy audit.
The original nondeterministic sparse selector and model execution are unchanged.

The full raw archives and replay inputs for all modes live in `artifacts`,
with member hashes re-read and verified. The archived store diagnostics include
all rank markers and explicitly avoid double-counting nested timing intervals.
