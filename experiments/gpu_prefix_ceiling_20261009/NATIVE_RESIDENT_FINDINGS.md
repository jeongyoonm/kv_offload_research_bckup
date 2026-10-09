# Background-store resident CPU reference

The allocation completed and stopped. Every measured request restored its
exact CPU prefix with zero GPU prefix hits. Native D2H is verified on all
eight ranks. Model execution, precision, cache budgets, predictor, 90 ms
subtraction, 100 ms setup lead and request seeds match sg2. Only native
D2H submission differs. Warm priming is excluded.

| Cohort | Req/s | Rate gain % | Mean TTFT ms | Pooled p95 ms |
| --- | ---: | ---: | ---: | ---: |
| familiar60k | 7.341 | +0.62 | 2926.6 | 4403.4 |
| short16k | 26.046 | +3.56 | 1052.7 | 1376.2 |

There are ten independently primed 32-request long-prompt waves and four
75-request short-prompt episodes, with concurrency 32. Independent
allocations and small repeat counts do not establish statistical
significance. The long cohort differs from earlier 75-prefix CPU runs.

`artifacts/cpu_native_resident` preserves the verified raw archive,
provenance, frozen loaded state, launch plan and derived comparison.
Serving acceptance is not a whole-model accuracy claim.
