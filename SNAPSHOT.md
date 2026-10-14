# Validated KV-offloading prototype backup

`main` contains the committed prototype from
`/mnt/localnvme/jeongyoon/vllm_cached_dispatch_horizon_20261014`, source commit
`acc46862a8c874a376c896489fa137d392016e27`. Prototype source files are copied
without modification. Backup documentation and `research_harnesses/` are added
alongside that snapshot.

The snapshot includes async chunked H2D/D2H submission, early KV-load submission,
worker preparation, scheduler hold/seal policies, compute deadlines, private CPU
cache initialization, safe store/block retirement, capacity reservations, and
the cached dispatch-depth fix. Experimental policies retain their opt-in defaults.
Alternative implementations and profiling experiments are preserved on the
175 original `experiment/*` branches at their original commits.

The prior backup main commit `5147414` remains on
`archive/main-before-prototype-20261014`. The main branch continues the backup
repository's independent snapshot history. The imported experiment branches
retain the source repository's available histories. Three additional
`experiment/tracelab-runner/*` branches preserve official-agent replay,
model-call-slot replay, and the HTTP-keepalive fix. Their original upstream
checkouts were shallow; that available-history boundary is preserved.

## Verification already performed on the source snapshot

- 617 CPU tests passed; 24 skipped.
- Four GLM5.3 TP8 TraceLab serving cells passed request, token, and transfer
  accounting checks, with original model selection and weights.
- The cached dispatch-depth fix improved throughput by 0.93% at concurrency 32
  and 2.54% at concurrency 48 against fresh controls with the same capacity and
  retirement policies. These are single comparisons, with mixed TTFT results.

The backup itself is verified by Git tree/blob comparison, branch-head equality,
file hashes, and repository object connectivity. It does not rerun GPU serving.

## Workload and replay harnesses

- Homogeneous and heterogeneous CPU hits:
  `experiments/load_only_overlap_20261011/`.
- Inferact preparation and causal replay:
  `experiments/agentic_1m_20261012/` and `experiments/real_traces_20261012/`.
- TraceLab official-agent replay, request-slot replay, pacing, concurrency,
  and capacity sweeps: the `experiments/tracelab_*_20261013/` directories.
- The most recent correctness/performance campaign:
  `experiments/cached_dispatch_horizon_20261014/`.
- External TraceLab Rust runner, CSV exporter, lockfile, licenses, frozen
  predictor/configuration inputs, and standalone launch helpers:
  `research_harnesses/`.

See `research_harnesses/README.txt` for entry points and external dependencies,
and `research_harnesses/MANIFEST.json` for source paths, commits, and hashes.
The scripts preserve their historical paths and timing parameters; rebind output
roots and allocation deadlines before starting a new campaign.

Model weights, raw traces, the enwik9 corpus, virtual environments, generated
token pools, compiled binaries, and large benchmark logs are external to this
source backup. Model weights remain at `/mnt/localnvme/huggingface/hub`.
