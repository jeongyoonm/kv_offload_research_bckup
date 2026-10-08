# Proportional load-only admission

Branch: `experiment/glm53-load-only-fraction-20261008`.
Checkout: `/mnt/localnvme/vllm_glm53_load_only_fraction_20261008`.
Artifacts: `/mnt/localnvme/glm53_load_only_fraction_20261008`.
Parent: `24c30df`, the latest frozen-predictor deadline comparison.

## Policy

Set `late_seal_load_only_ready_percent=50` in the full experiment options.
For a load-only predecessor, the readiness target becomes
`ceil(published_load_candidates * percent / 100)`: 16 requests need 8,
14 need 7, 9 need 5, and 1 needs 1.

The denominator is the bounded request cohort published for B, frozen at
publication. It counts request-level load candidates, not rank acknowledgments,
DMA chunks, or unrelated transfers outside B's candidate slots. Completed loads
remain in this denominator. All base ranks must acknowledge preparation; each
counted candidate must be KV-ready and preparation-acknowledged on every rank.
Sealing admits the eligible snapshot, which may contain more than the target.

Cancelled candidates do not contribute to the ready count. The original target
is capped at the number of surviving candidates so aborts cannot make it
unreachable. With an active compute predecessor, existing readiness targets,
compute deadlines, and completion exits apply. Safety expiry still bounds waits.

Zero disables the percentage rule and retains the existing absolute target.
A positive percentage overrides the absolute load-only and general targets
only in the load-only path. It requires mode 4 and either event sealing or
predictive sealing with prediction-to-readiness enabled. The old fixed-hold
partial-percentage option is independent of this rule.

## Integration

The option is validated, passed through engine configuration, and applied in
the readiness branch used by the latest predictive prototype. Transport traces
record `load_only_pending_candidates`, `load_only_ready_percent`, the frozen
`load_only_fraction_target`, and the effective `ready_target` after cancellation.

`profiles.json` contains full deadline-off and deadline-on options copied from
the completed 60K/16K/mixed comparison, with only the new percentage added.
Both use 50%; the active-compute target remains 16. Apply a profile's full
options through the existing experimental configuration API on this checkout.
Use the same frozen predictor referenced in that file. No previous run or
artifact was modified, and no GPU throughput evaluation has been performed.

The launcher reuses the complete parent experiment protocol and adds 50% in
both measured arms. It uses a fresh artifact directory and port 18412. Inspect
the generated options without launching a server:

```bash
.venv/bin/python experiments/load_only_fraction_20261008/run.py --print-profiles
```

When a GPU comparison is requested, run `run.py` without that flag with all
eight GPUs idle. Then use the parent CPU evaluation interpreter to run this
directory's `analyze.py`. Besides the original audit, the analyzer checks each
recorded load-only target against the frozen percentage and cancellations.

```bash
PYTHONPATH="$PWD" .venv/bin/python experiments/load_only_fraction_20261008/run.py
OPENBLAS_NUM_THREADS=1 \
  /mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/load_only_fraction_20261008/analyze.py
```

## CPU verification

Run from this checkout:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" .venv/bin/python -m pytest \
  tests/v1/kv_offload/test_late_admission_harness.py \
  tests/v1/kv_offload/test_early_readiness.py -q
```

Focused checks cover proportional thresholds in event/predictive modes, odd
counts and small cohorts, fixed-denominator behavior, eighth-rank candidate and
base acknowledgments, cancellation exclusion and liveness, active-compute
isolation, safety expiry, and option validation. Existing suites also exercise
ordered retirement, independent preparation, and deadline behavior.

Detailed outcomes and test logs are stored in the artifact directory above.

Focused percentage checks: **22 passed**, 583 deselected.
Broader CPU suites: **570 passed, 32 skipped, 3 failed**. The same three failures
reproduced on the unchanged parent checkout: an outdated timing-channel fixture
missing `seal_estimator`, a Path/string mismatch in preparation notification,
and a missing historical `opportunities.py` analysis artifact. These failures
were retained and were not caused by the percentage change. See the archived
verification record and logs; the full suite is not reported as passing.

Archived [verification](results/verification.json),
[focused checks](results/focused_tests.txt), [CPU suites](results/cpu_suite.txt),
and [parent reproduction](results/parent_failures.txt).
