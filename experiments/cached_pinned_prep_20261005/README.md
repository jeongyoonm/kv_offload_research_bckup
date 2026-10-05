# Cached requests and private pinned preparation

Integration branch: `experiment/glm53-cached-pinned-prep-20261005`, based on
coalesced notification prototype `0b1297d`. Artifacts are under
`/mnt/localnvme/glm53_cached_pinned_20261005`.

## Serving mechanism

Two independent, default-false experimental options are available:

- `late_cached_requests`: publish full immutable text-prefill data through the
  independent preparation lane. After all-rank acknowledgments, final execute
  carries a `(generation, publication step)` key and request identity instead
  of prompt lists. Final blocks, context and sampling fields remain authoritative.
- `late_pinned_packing`: materialize private pinned int32 CPU tensors in that
  lane. Ordered execute adopts selected tensors into the normal token writer;
  CPU-to-GPU copies and GPU writes happen after seal.

Both require late-admission mode 4 and the existing text-only restrictions.
Known base prefills are published too. Unacknowledged requests, ordinary-path
fallbacks and changed prompt data retain full execute payloads. An acknowledged
reference that cannot be resolved at worker entry fails closed. This is a bounded
cache for preparation packets, not a persistent prefix-KV cache.

The lane retains at most two preparation packets. Token writers retain at most
two copy-fence ownership groups; configuration reset drains these fences and
joins the old preparation lane before starting a new generation. Preparation
threads select their worker's accelerator before pinned allocation. Preparation
does not assign live request slots or upload token data to the GPU.

## Verification

Run focused CPU and explicitly enabled small GPU checks from the worktree:

```bash
export PYTHONPATH="$PWD"
export TMPDIR=/mnt/localnvme/glm53_cached_pinned_20261005
export VLLM_CACHE_ROOT="$TMPDIR/vllm-cache"
.venv/bin/python -m pytest -q \
  tests/v1/kv_offload/test_late_admission_harness.py
CUDA_VISIBLE_DEVICES=0 CACHED_PINNED_GPU_TEST=1 \
  .venv/bin/python -m pytest -q \
  tests/v1/kv_offload/test_late_admission_harness.py \
  -k 'cached_reference or pinned_preparation_adopts'
```

The real-model canary requires all eight GPUs idle and existing GLM weights and
compatible native/vendor dependencies. It uses port 18255, its own caches, and
all artifacts under the directory above. The inherited observer library is
unchanged. Training and correctness requests are excluded from measurements.

```bash
.venv/bin/python experiments/cached_pinned_prep_20261005/canary.py \
  --name gpu --policies current,references,cached_pinned \
  --workloads 8192:95:100,61440:95:100 \
  --requests 64 --repeats 2 --max-model-len 65536 --distinct-fact-probe
.venv/bin/python experiments/cached_pinned_prep_20261005/analyze.py
```

The three policies share coalesced notifications and the full prediction minus
50 ms / 2 ms polling policy. They compare the existing prototype, references
alone, and references plus pinned packing. The predictor is trained and frozen
before measured cells; policy order reverses for the second repeat. This is a
short correctness/performance canary, not a 300-request comparison.

## Repeated 60K regression check

`CACHED_PINNED_ARTIFACT_ROOT` selects a fresh artifact directory without
overwriting the original canary. The follow-up uses six paired repeats of
80 requests at concurrency 32 and alternating policy order. Its predeclared
practical throughput regression margin is 2%. The confidence interval uses
paired log throughput ratios and describes variation for this workload only.

```bash
export CACHED_PINNED_ARTIFACT_ROOT=/mnt/localnvme/glm53_cached_pinned_60k_regression_20261006
.venv/bin/python experiments/cached_pinned_prep_20261005/canary.py \
  --name gpu --policies current,cached_pinned --workloads 61440:95:100 \
  --requests 80 --repeats 6 --max-model-len 65536 --distinct-fact-probe
.venv/bin/python experiments/cached_pinned_prep_20261005/analyze.py
.venv/bin/python experiments/cached_pinned_prep_20261005/check_60k.py \
  "$CACHED_PINNED_ARTIFACT_ROOT"
```

The analyzer also separates request-state updates, input preparation, attention
metadata, and the interval from metadata completion to forward submission.
Each phase reports the slowest rank per prefill dispatch. These are host wall
times and can include CUDA or connector waits; they are not isolated CPU costs.
