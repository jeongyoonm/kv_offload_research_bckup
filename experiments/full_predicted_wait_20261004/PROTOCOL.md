# Full predicted initial wait: GLM 5.3 TP8

Opt-in branch: `experiment/glm53-seal-full-predicted-wait-20261004`.
Artifacts: `/mnt/localnvme/glm53_full_predicted_wait_20261004`.

The initial wait uses the first conditioned predicted seal deadline minus a
margin of 50 or 0 ms, without a 50 ms per-wake cap. Progress interrupts the wait;
the remaining initial wait resumes toward the same fixed transition. After
that transition, poll at 2, 5, or 10 ms. The forecast never seals a batch.
Readiness, physical compute completion, cancellation, and the existing 2-second
safety limit determine membership. Ordinary preparation and ordered retirement
remain authoritative. Missing forecasts use fine polling. Load-only batches
keep 2 ms polling and the existing four-candidate target in every full-wait cell.

The existing transports publish memory rather than native wake notifications.
An experiment-only read-only bridge checks ring heads, execution completion
words, preparation ACKs, and cancellation notices every 1 ms during coarse
waits. Its notifications are advisory; it never consumes KV completion rings,
polls RPC responses, or changes scheduler state. The engine revalidates all
conditions. A generation counter prevents a notification between snapshot and
wait from being lost. Completed compute continues notifying until the engine
sees an authoritative result, including temporary timing-reader contention.
The observer is paused during fine polling and drained before sealing.
Its full thread CPU time, scans, and notifications are recorded separately.
This implementation shifts lightweight observation to a background thread;
it is not polling-free and its cost must be included in any CPU comparison.

Controls are the previous capped-50-ms hybrid with 2 ms fine polling and
readiness-only 2 ms polling. Every policy records individual wait boundaries,
requested duration, phase, wake reason, and preceding progress iteration start.
These distinguish sleeping after a stopping condition from later progress work.

Use the established matched workload matrix: 8192 tokens / 95% cached (300
requests), 61440 / 95% cached (80), and 61440 / 99% cached (80). All requests hit
the independently primed CPU prefix cache; concurrency is 32, output length 1,
candidate slots 10, FP8 KV, worker chunks 128 MiB and pipeline depth 2. Long
workloads retain 80 requests to fit the physical CPU cache. Run all eight
policies with seed 57, then repeat with seed 58 in reversed policy order.
Predictor observations remain past-only and are retained across policies, as
in the previous study. Every cell resets and primes its cache independently.

CPU barriers verify interruptible compute completion while A's RPC remains
held, candidate readiness, cancellation, notification-before-wait races, and
transition to each fine interval. Known-answer and distinct-fact probes precede
performance replays. Reject cells with cache mismatch, preemption, GPU cache
hits, measured JIT compilation, output-count mismatch, or unexpected processes.
Retain throughput, mean/p95 TTFT, admissions, compute rounds, exposed-hold and
response-delay diagnostics, main polling CPU and bridge CPU. No policy is
promoted based on a single replay.
