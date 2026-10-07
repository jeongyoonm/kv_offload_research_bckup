# Offset-only duration prediction and live accuracy audit

Branch: `experiment/glm53-offset-only-predictor-20261008`.
Checkout: `/mnt/localnvme/vllm_glm53_offset_predictor_20261008`.
Artifacts: `/mnt/localnvme/glm53_offset_predictor_20261008`.

Predict total critical completion duration without elapsed execution time.
The scheduler retains its absolute GPU-start-based deadline and 5 ms lead.
Elapsed time reduces the remaining wait through subtraction. Checkpoints for
this opt-in experiment mask column 21 during both fitting and inference,
including the outside-support fallback. The column remains in diagnostics and
causal snapshot validation. Other transfer progress features are retained.

## Controls

- Match tree and MLP capacities, training rows, weights and seeds between
  elapsed-input and offset-only models.
- Fit on the existing original pool, then separately add previous live decision
  snapshots. This distinguishes feature removal from better data coverage.
- Purge all predictor variants sharing a workload and request seed from each
  held fold. Four folds hold each complete repeat across all three workloads.
- Select the offset architecture on grouped validation only. Freeze both
  matched checkpoints before new live requests. The original 192-row replay
  benchmark is excluded from fitting. Prior live data are now development data.
- Compare current serving trees, matched elapsed-input and matched offset-only
  models on fresh seeds 2301–2304. Concurrency is 32, output length is one token.
  Each predictor/workload receives four 75-request episodes, totaling 300.
- Preserve GLM 5.3 NVFP4 TP8, FP8 KV, private CPU cache initialization, chunked
  async H2D, optimized compact/pinned preparation, and the original selector.
- Prime/reset separately per cell; exclude startup, priming and nine warmups.
  A two-hour watchdog bounds the server allocation and stops its process group.

## CPU checks

Use `/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python`
for `study.py`, `verify.py`, `audit.py`, `conditions.py`, and `analyze.py`.
Serving and pytest use this checkout's `.venv/bin/python`.

`study.py` records fitted checkpoints, fold predictions, source weighting,
selection and export hashes. `--seed 74` and `--seed 75` write independent
subdirectories; these sensitivity checks do not change the frozen live models.

`verify.py` checks export equivalence, snapshot restoration, causality, and
elapsed independence of both the primary model and fallback. The existing
predictor pytest suite also contains small elapsed-sensitive trees to verify
that masking applies throughout both serving paths.

## Audit

`audit.py --root ARTIFACT_ROOT --output AUDIT_DIRECTORY` reproduces every issued
prediction from its checkpoint and saved inputs, verifies labels/deadlines,
replays every model on common snapshots, and measures elapsed sensitivity with
other state fixed. `conditions.py` compares transfer and submission conditions.
New live traces retain the exact rank snapshot used at prediction time.

The target is the CUDA event span through critical completion. It includes
host submission gaps and required output-copy joins. CPU submission end is a
diagnostic label and is never supplied when it occurs after prediction time.
Sensitivity changes one feature, not actual GPU execution. Distribution shifts
and correlations do not establish their individual causal contributions.

Run serving after CPU export checks:

```bash
PYTHONPATH="$PWD" .venv/bin/python experiments/offset_predictor_20261008/run.py
```

Frozen-model, rank-preparation, CPU restoration, scheduled-token, GPU-prefix-hit
and preemption checks are required for measured cells. Raw traces remain under
the artifact root; compact reports and final evidence are copied into this
branch when the comparison completes.
