# Synchronous-store resident CPU reference

The allocation completed and stopped. Warm priming is excluded. Every
measured request restored the exact CPU prefix with zero GPU prefix hits.
The model, FP8 KV precision, 96 GiB GPU pool and 2.13 TB private CPU pool
retain their prior settings. Allocator avoidance remains enabled.

| Cohort | Requests | Req/s | Mean TTFT ms | Pooled p95 ms |
| --- | ---: | ---: | ---: | ---: |
| 60K / 32 prefixes | 320 | 7.296 | 3002.2 | 4382.5 |
| 16K / 75 prefixes | 300 | 25.151 | 1088.8 | 1428.4 |

There are ten independently primed 32-request long-prompt waves and four
independently primed 75-request short-prompt episodes, at concurrency 32.
The long cohort is distinct from the earlier 75-prefix CPU workload; their
absolute rates cannot establish a cache-placement effect. Serving
acceptance is not a whole-model accuracy claim.

The full raw archive and member hashes are verified in
`artifacts/cpu_synchronous_resident`. Its source commit is recorded there.
The driver checks the frozen checkpoint hash before startup, but its
original plan omitted the file hash; the final audit records that evidence
explicitly and checks the loaded runtime-state hash.
