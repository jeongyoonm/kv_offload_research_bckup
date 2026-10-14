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

## Preparation contention diagnosis (2026-10-06)

Diagnostic branch: `experiment/glm53-preparation-contention-20261006`, based on
the measured batched-transfer implementation `17c9739`. Full artifacts, including
the Nsight Systems report and exported SQLite trace, are under
`/mnt/localnvme/glm53_prep_contention_20261006`.

`contention_probe.py` drives the actual independent preparation lane with ten
60K text prompts while one GPU receives a fixed sequence of eager kernels or
CUDA graph replays. It measures the lane mutex and adoption call. Notification
publication is replaced with a no-op to isolate decoding, conversion and pinned
allocation; this is not a serving benchmark. Twelve warmed alternating repeats
are used per mode. A separate three-repeat Nsight run traces CUDA and Python GIL
ownership. Its absolute timings are not substituted for unprofiled results.

The direct trace shows the main submission thread waiting for the GIL while the
preparation thread owns it. In graph mode, full pinned preparation produces about
25ms of main-thread GIL wait per packet. CUDA launch API durations remain short;
the preparation thread issues no device/stream synchronization. Warm pinned
allocation takes about 0.09ms per packet and lane-mutex waits remain below 0.01ms
in this probe. Reusing pinned tensors does not remove the submission delay.

The unprofiled graph GPU span increases from 71.64ms without preparation to
92.67ms with full pinned preparation. If GPU work is queued before preparation
starts, the same span stays at 71.64ms, with every treatment verified to begin
while GPU work remains pending. This separates host-submission interference from
slowing already queued GPU kernels in the tested sequence.

A separate decode-only treatment publishes NumPy arrays in the packet instead
of Python token lists. Within that matched run, array decoding adds about 0.07ms
to the GPU span versus about 11.49ms for list decoding. This is a representation
experiment, not an integrated replacement for the cached-request wire contract.

The saved TP8 traces also show preparation completion before A's worker step has
finished submitting in 528 of 552 rank/window instances. That boundary includes
sampling and output submission; it does not locate each individual model kernel.
GIL contention is therefore a demonstrated mechanism with a relevant serving
overlap window, but its share of the observed 1.55% throughput loss remains
unmeasured. KV-transfer contention, changed batch shapes and ordered writer-fence
waits require a matched model trace to separate.

Reproduction (idle GPU 0; no model weights are loaded):

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD
export TMPDIR=/mnt/localnvme/glm53_prep_contention_20261006
.venv/bin/python experiments/cached_pinned_prep_20261005/contention_probe.py \
  --root "$TMPDIR/probe" --repeats 12 --launches 2048
.venv/bin/python experiments/cached_pinned_prep_20261005/contention_probe.py \
  --root "$TMPDIR/prequeued" --repeats 12 --launches 2048 \
  --launch-modes graph --prequeue
```

Committed compact summaries are in `results/contention_20261006`. Production
serving code and default flags are unchanged in this diagnostic branch.

## Compact preparation packets (2026-10-06)

Branch: `experiment/glm53-compact-preparation-tokens-20261006`.
`late_compact_tokens` defaults to false and requires the existing mode-4 cached
or pinned preparation option. The engine validates immutable text prefills and
encodes tokens once into contiguous int32 bytes. Worker decoding retains small
request metadata and creates read-only NumPy views without reconstructing Python
token lists. Selected cached references resolve to these arrays or their prepared
pinned copies privately in the V2 runner. Final allocation and accounting still
come from the sealed scheduler output. Unselected buffers and generation resets
use the existing bounded ownership and drain paths.

The focused interference probe uses both packet formats in the same run, with
twelve alternating warmed repeats. Graph GPU spans are 71.54ms for idle, 93.33ms
for the old pinned packet, and 71.55ms for compact pinned preparation. The CPU
harness passes 146 cases; eight explicit GPU adoption cases cover 8K/60K prompts,
selected slot ordering and changed-prompt fallback against ordinary runner state.

The matched GLM5.3 TP8 comparison uses 60K prompts, 95% CPU prefix lengths, 100%
CPU hits, concurrency 32, one output token, and three alternating repeats of
80 requests per policy. `current` is the existing complete prototype, without
cached references or pinned preparation; `cached_pinned` is the measured earlier
optimization; `compact_pinned` adds this wire-format fix. All policies share a
trained frozen predictor and coalesced notifications. Known-answer and distinct
fact correctness checks are rerun. Model weights and KV precision are unchanged.

```bash
export PYTHONPATH=$PWD
export CACHED_PINNED_ARTIFACT_ROOT=/mnt/localnvme/glm53_compact_prep_20261006
.venv/bin/python experiments/cached_pinned_prep_20261005/canary.py \
  --name gpu --policies current,cached_pinned,compact_pinned \
  --workloads 61440:95:100 --requests 80 --repeats 3 \
  --max-model-len 65536 --distinct-fact-probe
.venv/bin/python experiments/cached_pinned_prep_20261005/analyze.py
.venv/bin/python experiments/cached_pinned_prep_20261005/compare_compact.py \
  "$CACHED_PINNED_ARTIFACT_ROOT"
.venv/bin/python experiments/cached_pinned_prep_20261005/verify_compact.py \
  "$CACHED_PINNED_ARTIFACT_ROOT"
```

All nine measured cells were accepted: 720 requests, with the same 245,760 suffix
tokens and 14 compute rounds per cell. All-rank coverage shows 239 of 240 requests
adopted compact pinned state on each rank. Source hashes, unique request IDs,
frozen settings, CPU sampler health, timing coverage, expected CPU hits and equal
KV bytes were verified. There were no GPU prefix hits, preemptions, measured JIT,
runtime errors or foreign GPU processes. Every policy passes 16/16 known-answer
and 16/16 distinct-fact checks. Exact distinct-fact text matches against the cold
reference are 14/16 for current, 16/16 for cached/pinned and 14/16 for compact;
the factual answers all match.

| Policy | Mean req/s | Mean TTFT (ms) | Mean cell p95 TTFT (ms) |
| --- | ---: | ---: | ---: |
| Existing prototype | 7.072 | 3945 | 5783 |
| Cached references + batched pinned transfer | 6.954 | 4008 | 5428 |
| Compact tokens + cached/pinned | 7.122 | 3915 | 5683 |

Compact improves the mean throughput by 2.41% against old cached/pinned and 0.71%
against the existing prototype. The latter paired 95% interval is -0.62% to
+2.06%; against old cached/pinned it is -1.05% to +6.00%. These three-repeat
intervals include no gain, while excluding a 2% slowdown against current. The
observed mean throughput gap is closed on this workload. Mean TTFT improves by
2.30% against old cached/pinned and 0.75% against current. Compact p95 improves
against current, but is 4.70% higher than old cached/pinned.

Mean summed GPU execution spans per cell change from 10.806s for old cached/pinned
to 10.536s for compact, versus 10.437s for current. Mean intercompute gap bounds
are 475–488ms, 493–505ms and 677–689ms respectively. Compact removes about 270ms
of the old path's execution-span inflation while retaining shorter gaps than
current. The spans include host submission, connector work and output
synchronization; they are not isolated kernel times, and batch membership shapes
can also affect them.

Full artifacts are under `/mnt/localnvme/glm53_compact_prep_20261006`; compact
evidence is committed in `results/compact_20261006`. The measured implementation
is `e45616d`. The new option remains disabled by default.

## Compute seal deadline follow-up (2026-10-06)

Branch: `experiment/glm53-seal-deadline-20261006`; serving commit `bcc4212`.
Artifacts: `/mnt/localnvme/glm53_seal_deadline_20261006`.

The new `late_seal_compute_deadline` option defaults to false. It requires the
existing predictive readiness policy and arms one fixed deadline at predicted
preceding execution completion minus `late_seal_compute_deadline_lead_us`.
The effective expiry is the earlier of this deadline and the existing maximum
wait. Candidate readiness and physical completion can still seal earlier.
Missing prediction retains the maximum-wait fallback. All base acknowledgments
remain required; load-only decisions retain the existing policy. Both coarse
notification waits and fine polling sleeps are clipped to the effective expiry.
`late_seal_compute_deadline_shadow` records the deadline without enforcing it;
shadow and active deadline modes are mutually exclusive.

The CPU harness passes 174 checks, including deterministic snapshot boundaries,
partial acknowledgments, cancellation, late base preparation, missing prediction,
early completion, load-only behavior, and execute enqueue before held A release.
Real notification waits exercise both the bridge and coalesced transport. Eight
GPU adoption checks pass. All five TP8 canary settings pass 16/16 known-answer
and 16/16 distinct-fact checks.

The canary uses the same 60K workload and frozen predictor for one 80-request
cell per setting. Control means the complete compact/pinned readiness prototype
with shadow logging, without deadline enforcement.

| Lead | req/s | Mean TTFT ms | p95 TTFT ms | Compute rounds |
| --- | ---: | ---: | ---: | ---: |
| Control | 7.028 | 3978 | 5371 | 14 |
| 0ms | 7.094 | 3923 | 5720 | 14 |
| 5ms | 7.304 | 3831 | 5290 | 14 |
| 10ms | 7.054 | 3990 | 5744 | 14 |
| 20ms | 7.238 | 3830 | 5045 | 13 |

Deadline overshoot is 0.11–0.17ms in the canary. These single cells establish
mechanism coverage and provide selection evidence, not a repeatable speedup.
Static shadow replay of the preceding compact experiment would miss 15 of 206
actual late admissions even at zero lead; it does not model later batching
feedback. Actual canary admissions therefore need to be measured independently.

The 5ms lead advances to four alternating paired 75-request episodes, totaling
300 measured requests per policy, under `comparison_300`. Cache resets and
priming occur between episodes and are excluded. This retains the 60K workload
within the existing CPU cache; it is not a continuous 300-request stream.
`experiments/seal_deadline_20261006/analyze.py` audits serving traces and
`compare_pairs.py` reports a paired interval from the four episode ratios.

The four paired episodes are complete and accepted. Control is 7.107 req/s and
5ms lead is 7.193 req/s (+1.21%); mean TTFT improves 1.26%, while mean cell p95
worsens 0.71%. The paired throughput 95% interval is -0.69% to +3.15%. Compute
rounds rise from 50 to 52 and late admissions fall from 280 to 276. The feature
remains disabled by default. Full results and scope are in
`experiments/seal_deadline_20261006/README.md` and its `results/` directory.
