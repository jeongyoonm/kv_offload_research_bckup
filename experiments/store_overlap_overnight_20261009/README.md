# Store overlap overnight exploration

Start: 2026-10-09 05:47:40 UTC. Budget: ten hours.
Model: GLM 5.3 NVFP4, TP8 on B300 GPUs.
Data: `/mnt/localnvme/jeongyoon/store_overlap_overnight_20261009`.
See `plan.json` for branch, source, artifact, and verification provenance.

The prior GPU stream fence removed CPU waits but did not demonstrate throughput
gain. It reduced inter-compute gaps while moving some waiting onto GPU streams;
initial request/load arrival also varied before the changed fence executed.
This exploration first distinguishes compute, future loads, reserved suffix
blocks, and allocator choices. It preserves real store completion before any
source overwrite and exactly-once all-rank retirement. Each proposal has its
own branch; negative and failed experiments remain part of the evidence.

Use short temporary/artifact paths below the requested storage root: the first
long path failed FlashInfer's Unix socket initialization and was rejected before
measurement. GPU-only controls must verify resident prefix capacity and actual
hit rates. A 75-request set of distinct 60K prefixes exceeds the fixed GPU pool;
capacity-fitting controls will be labeled separately from full serving episodes.
