# Matched capacity-fitting cache comparison

GLM 5.3 NVFP4 TP8, FP8 KV, concurrency 32 and one output token. GPU KV is
96 GiB per rank in all modes. CPU modes use the same 2.13 TB private cache,
frozen predictor and scheduler policies. Warm priming is excluded.

| Cache / store mode | 60K req/s | 16K req/s |
| --- | ---: | ---: |
| CPU / synchronous | 7.296 | 25.151 |
| CPU / background native | 7.341 | 26.046 |
| GPU resident | 7.720 | 25.770 |

The worker change improves matched CPU throughput by 0.62% and 3.56%.
Mean TTFT falls from 3002 to 2927 ms on 60K and 1089 to 1053 ms on 16K.
Pooled p95 changes from 4382 to 4403 ms and 1428 to 1376 ms respectively.
GPU resident mean TTFT is 3066/1091 ms; its p95 is 4121/1256 ms.
These are descriptive independent allocations, not confidence intervals.

Every measured CPU request restores its complete expected prefix with zero
GPU prefix hits. Every GPU request hits its complete expected prefix with
zero CPU offloading loads. THP disabling is audited on all GPU workers;
CPU runs retain their existing process-local THP and NUMA policy. Original
model selection and execution remain unchanged.

The long cohort has ten independently primed waves of 32 distinct prefixes,
320 measured requests. The short cohort has four independently primed
75-request episodes, 300 measured requests. Their aligned prefix lengths
are 58,368 and 15,552 tokens; prompt lengths are 61,440 and 16,384. Each
cohort matches seeds and request counts across modes.

Improved CPU throughput reaches 95.1% of GPU resident on the long cohort
and 101.1% on the short one. GPU-only removes connector-dependent scheduling
and uses ordinary preparation; it is a measured reference, not a strict
upper bound. The short result does not prove CPU transfers accelerate
computation. Separate scheduling and arrival effects remain possible.

The original 75-prefix long working set exceeds the fixed GPU budget. Its
capacity-limited GPU rate, 0.494 req/s, is preserved separately and cannot
serve as an all-hit ceiling. The earlier unaudited THP reference is also
retained separately. Use sg4 for the matched resident comparison.

Raw results and verified archives reside on the three dedicated reference
branches. The data report includes all repeat measurements and metrics.
