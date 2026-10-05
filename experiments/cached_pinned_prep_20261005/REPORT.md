# Cached request references and private pinned packing integration

## Scope and implementation

Branch: `experiment/glm53-cached-pinned-prep-20261005`, based on `0b1297d`.
Measured serving implementation: `c478aeb`; driver fixes: `6e7c10c`.
Full artifacts: `/mnt/localnvme/glm53_cached_pinned_20261005`.
The parent worktree was not modified.

The existing independent worker preparation lane now caches immutable request
lists and packs token IDs into private pinned CPU tensors. All-rank acknowledgments
allow final execute to carry request references instead of full prompt lists.
The final SchedulerOutput retains authority over membership, allocation and
context. Ordered execution resolves the references and adopts selected packing.
GPU uploads still occur after sealing. Both experimental flags default to false.
Preparation packets and copy-fence ownership are bounded and drained on reset.

A final hardening change discards prepared state if an authoritative full payload
has changed prompt data. Missing or stale acknowledged references still fail closed.
This fallback was tested separately after the measured serving run; it is not exercised
by the immutable prompts in the performance cells.

## Serving comparison

GLM 5.3 NVFP4, TP8, B300 GPUs, FP8 E4M3 KV. Concurrency 32; 64 requests per
cell; two repeats, policy order reversed on repeat two. Prompt lengths 8192 and
61440, 95% prefix length, 100% CPU prefix hit target, GPU prefix hits zero.
All policies share the current full prediction-minus-50ms / 2ms polling policy,
coalesced notifications, ten candidate slots, two-chunk asynchronous H2D and
earlier KV submission. One common predictor was trained then frozen.
The control is the existing full prototype, not unoptimized vLLM.

<!-- markdownlint-disable MD013 -->

| Prompt | Configuration | Req/s | Throughput gain | Mean TTFT (ms) | TTFT reduction | p95 TTFT (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8K | Current full prototype | 32.326 | +0.00% | 835.7 | +0.00% | 1114.2 |
| 8K | + cached request references | 34.027 | +5.26% | 800.4 | +4.23% | 1086.2 |
| 8K | + references and private pinned packing | 35.080 | +8.52% | 772.1 | +7.61% | 1059.2 |
| 60K | Current full prototype | 6.910 | +0.00% | 3856.2 | +0.00% | 5533.0 |
| 60K | + cached request references | 6.855 | -0.79% | 3878.6 | -0.58% | 5572.8 |
| 60K | + references and private pinned packing | 6.846 | -0.92% | 3894.4 | -0.99% | 5424.9 |

<!-- markdownlint-enable MD013 -->

The combined 8K improvement was positive in both repeats (+8.9%, +8.1%).
Reference-only repeats had substantial variation. Neither new configuration
produced a meaningful throughput benefit at 60K in this canary.
These short measurements establish mechanism feasibility, not statistical
confidence or general workload speedup. No 300-request comparison was run.

The 8K combined mean of per-cell median prefill CPU setup times decreased from
11.38 to 8.53 ms; at 60K it increased from 51.97 to 58.18 ms. Batch membership
also differs (8K mean compute rounds 10.5 versus 9.5; 60K 10 versus 10.5),
so the setup and throughput differences cannot be attributed solely to packing.
Dispatch-to-worker and dispatch-to-GPU traces include FIFO wait behind A and
must not be interpreted as isolated RPC decoding time or exposed GPU bubbles.

All twelve accepted cells have expected CPU prefix hits, zero GPU prefix hits,
zero preemptions, unchanged transferred bytes, no runtime errors, and no
measured compilation. All eight worker exports verify reference use in both
new policies and pinned packing use in the combined policy. One initial 8K
combined cell encountered a new Triton metadata-alignment specialization;
the guard rejected it and an accepted warm retry replaces it in this report.

## Correctness and verification

- Independent CPU harness: 138 passed; four explicitly gated GPU checks skipped.
- Focused integration: 10 CPU checks plus two real GPU runner checks passed.
- Final changed-prompt hardening: 138 CPU harness checks and four actual GPU
  runner cases passed, including ordinary full-payload fallback at both lengths.
- Real GLM known answers: 16/16 per policy, all three policies.
- Distinct facts: 16/16 correct and 16/16 exact cold-output matches per policy.
- GPU runner checks exercise 8192/61440 tokens, reordered selected requests,
  unselected slots, private CPU-only preparation and copy-fence draining.
- Broader regression suite: 499 passed, 14 skipped, three failures. All three
  reproduce on unmodified parent: an old timing fixture lacks seal_estimator,
  existing Path-plus-string handling, and an absent opportunities module.

The first serving attempt passed all model correctness checks but AIPerf could
not bind its Unix IPC socket under the long artifact TMPDIR. No performance
requests were issued. The retry uses a short task-local TMPDIR. Serving source
hashes matched; correctness summaries were reused with provenance checks.
The actual launch manifest records the isolated port 18255 and cache locations.

Summaries, launch provenance, test logs and compressed diagnostic traces are
saved in `results/`; full prompt and profile artifacts remain at the root above.
All owned server processes exited and the GPU inventory was empty after cleanup.
The parent worktree remained clean. All final staged pre-commit checks passed.
See README.md for reproduction commands. AI assistance: Codex.
