GLM5.3 TP8 load-only timing and safe H2D experiment, 2026-10-11

Storage: /mnt/localnvme/jeongyoon/load_only_overlap_20261011.
Model weights remain /mnt/localnvme/huggingface/hub.
Original sparse-attention selector, model weights and numeric execution unchanged.
Private CPU-cache initialization is shared by every comparison arm.

Serving controls
- Existing optimized worker: native H2D 8MiB, depth2; native D2H1GiB,
  allocator preference to avoid live store sources, and safe store reuse fences.
- H2D proposal: destination-specific compute/zero/COW fences on a private
  dependency stream. Conflicting stores remain protected. Default disabled.
- Load-only proposal: the same readiness target whether an empty preceding RPC
  has already returned or returns before base preparation acknowledgments.
  Idle targets1/4/8 cap the existing 50% candidate-ready target. Ordinary useful
  base work seals immediately; compute predecessors retain existing fallbacks.
- Full prototype retains independent preparation, final membership snapshot,
  candidate16slots, frozen_A predictor, 50ms deadline lead, original waiting
  and other policy choices. The serving plan stores exact per-arm fields.

Correctness vs performance
- Native tests check real copies: unrelated compute overlaps; destination
  readers, zeroing and stores preserve ordering and exact bytes.
- AuditedOffloadingConnector adds synchronous GPU-to-CPU validation, descriptor
  checks, CPU fingerprints and pre-forward validation. It is never enabled in
  performance runs. Preparation coverage and timing can be distorted by it.
- audit_gate checks byte evidence separately; it does not rewrite the rejected
  performance-cell acceptance or claim preparation overlap from that audit.
- Production throughput cells still require preparation coverage for full arms,
  exact restored prefix lengths, one output token, no preemptions or GPU hits,
  and identical frozen predictor state.

Broader workload measurements
- Homogeneous grid: cached prefix4096/16384/32768/65536 tokens by uncached
  prompt suffix128/512/2048/8192 tokens. Output length is one token.
- Max model length81920 for every broader arm; max batched tokens61440 and
  max sequences64, GPU KV96GiB/rank, CPU2130300354560bytes server-wide.
- Baseline uses ordinary host H2D/D2H with no chunked native submission,
  early submission, cooperative progress or scheduler late admission.
- Worker uses optimized I/O, early submission and store/allocator safeguards,
  plus safe block readiness. Full additionally enables preparation/hold/seal.
- Independent CPU-hit episodes are at most64 requests so primed prefixes fit
  the fixed CPU budget. Every episode resets caches, cold-primes prefixes,
  drains write-back, then resets GPU cache before measured requests. Cold
  priming, shape warmup, post-episode drain and reset are outside latency and
  throughput measurement. These are bounded CPU-hit episodes, not a
  continuously replenished steady-state run at concurrency64.
- A continuous evolving-cache cell warms a192/384 prefix working set, then
  retains both caches through its measured stream. Hot/cyclic reuse and new
  prefixes allow normal GPU promotion, GPU/CPU eviction and write-back.
  Intended reuse class is not a realized hit. Measure actual completed loads,
  cached token usage, GPU/CPU hits/misses and completed transfer bytes.
- CPU hit: any CPU load for that request; GPU hit: cached tokens>0 without
  CPU load; miss: neither. Requests with partial GPU+CPU hits count as CPU
  hits. Transfer bytes are reported per rank and across all eight ranks.
  These are serving-path outcomes, not an arrival-time residency oracle;
  requests sharing a pending promotion can reuse GPU blocks later.
- Initial engine-visible cohort barrier excludes frontend arrival skew before
  the first schedule. Subsequent arrivals are ordinary closed-loop arrivals.
- Shared compressed token datasets and hashes are recorded for matching arms.
- Current late candidate admission requires residual suffix<=4096 tokens.
  The8192-suffix cells therefore also test a limitation of the existing path;
  larger candidate support is not silently enabled for this evaluation.
- Heterogeneous classes are selected from measured transfer and compute demand,
  with explicit named class ordering for equal and transfer-skewed mixtures.

Coverage and reporting
- campaign.py preserves the active baseline evaluation and replaces only the
  stopped queue controller after its completion. Every subsequent launch is
  gated on the previous server stopping and every expected cell being accepted.
- coverage_plan.json records budget decisions explicitly. The dense sweep
  covers all seven concurrency points for three profiled classes; other shapes
  initially cover1/32/64, with16 added only if measured wall time allows.
- The64-request concurrency64 point is a single burst. It does not establish
  a continuously replenished steady-state throughput plateau.
- report_diverse checks matched payload hashes and budgets, observed transfers
  against both per-rank totals and Prometheus counters, and all-rank stores.
  Worker preparation trace coverage is separate from native execution timing;
  absent preparation markers are never counted as prepared inputs.
- Mixed-workload bars show repeat means and observed ranges, not confidence
  intervals. CPU submission and GPU execution envelopes overlap in time and
  must not be added as independent costs.
