# Load-only readiness and predictor deadline factorial comparison

GLM 5.3 NVFP4, TP8, concurrency 32, one output token. The four arms are
`all_off`, `half_off`, `all_on`, and `half_on`: wait for all candidates versus
50% readiness in load-only periods, each with the compute sealing deadline
disabled or enabled. The disabled deadline is traced in shadow mode. All arms
use the same frozen offset predictor, preparation, and H2D optimizations.

| Workload | Context tokens | Uncached suffix tokens | CPU prefix restoration |
| --- | --- | --- | --- |
| 60K | 58,368 | 3,072 | All 75 requests |
| 16K | 15,552 | 832 | All 75 requests |
| Mixed | 4,096 cached / 24,576 cold | 2,048 or 8,192 | 38 of 75 requests |

Each workload has four independently reset and primed 75-request episodes per
arm (300 measured requests per arm). Workloads are 60K cached, 16K cached, and
mixed context/suffix/cache status. The four-arm Williams order balances position
and immediate carryover across repeats. Prompts and seeds match within blocks.
Twelve 32-request warmups, startup, and priming are excluded. Confidence
intervals use episodes as the sample unit; four blocks provide limited precision.

The fractional target uses the frozen published candidate cohort, up to 16
slots, and counts both KV readiness and all-rank preparation acknowledgment.
Canceled candidates cap the target; completed candidates retain the denominator.
This does not change the rule while preceding compute is active. Original
sparse-attention selection is preserved. Private CPU cache initialization uses
the existing 2,130,300,354,560-byte aggregate budget.

Serving implementation is inherited unchanged from `673ddfa`; its CPU checks
reported 22 focused passes and 570 broader passes, with 32 skips and three
previously reproduced failures on the unchanged parent. This evaluation only
adds experiment orchestration and analysis.

Artifacts: `/mnt/localnvme/glm53_load_only_deadline_ab_20261008`.

```bash
PYTHONPATH="$PWD" .venv/bin/python \
  experiments/load_only_deadline_ab_20261008/run.py
```

Audit and analysis use the existing CPU evaluation environment with SciPy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/load_only_deadline_ab_20261008/analyze.py
```

`--audit-only` audits completed episodes without requiring the final cell count.
The audit verifies strictly ordered unique retirement, exact scheduled tokens,
CPU restoration counts, matching transfer volume and seeds within each block,
all-rank execution timing and preparation coverage, frozen predictor replay,
deadline arithmetic, and readiness targets. Partial-cohort KV completions are
compared with recorded compute spans, which include CPU submission stalls and
output joins. This is a serving protocol check, not a model quality evaluation.
