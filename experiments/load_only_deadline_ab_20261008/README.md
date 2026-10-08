# Load-only readiness and predictor deadline factorial comparison

GLM 5.3 NVFP4, TP8, concurrency 32, one output token. The four arms are
`all_off`, `half_off`, `all_on`, and `half_on`: wait for all candidates versus
50% readiness in load-only periods, each with the compute sealing deadline
disabled or enabled. The disabled deadline is traced in shadow mode. All arms
use the same frozen offset predictor, preparation, and H2D optimizations.

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
