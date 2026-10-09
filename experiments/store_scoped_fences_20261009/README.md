# Scoped store reuse fences

Branch: `experiment/store-reuse-scoped-fences-20261009`.
Parent: `6aeab52` (global GPU event fence diagnosis).
Artifacts: `/mnt/localnvme/jeongyoon/store_overlap_overnight_20261009/scoped`.

Hypothesis: future asynchronous H2D destinations can unnecessarily gate current
compute when all reuse dependencies are recorded on the compute stream.
Classify collisions at final scheduler metadata construction. Keep compute
collisions on the compute stream; submit each affected H2D load through a private
CUDA gate that preserves its store-before-overwrite dependency. Unclassified
lifecycle flushes remain global. Reset, synchronous-load and block-mutation
batches retain the broader path. Stores retire only after real all-rank completion.
Both experiment options default to false.

Tests guard against source corruption, dropping dependencies that touch both
compute and load, accidentally fencing unrelated GPU work, and duplicate stores.
Matched serving runs toggle only scoped fences, with the previous global GPU
fence as control. Model, cache capacity, predictor and scheduler policy are fixed.

The overnight task began 2026-10-09 05:47:40 UTC and is budgeted for ten hours.
Next proposals depend on this result: pending-store allocator preference,
critical-store submission ordering, then bounded D2H copy chunks. Every proposal
will have a separate branch and preserve negative results.
