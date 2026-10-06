# Architecture comparison run log

## Setup and protocol

- Worktree: `/mnt/localnvme/vllm_glm53_predictor_architecture_cpu_20261007`.
- Branch: `experiment/glm53-predictor-architecture-cpu-20261007`.
- Parent: `9d301ca`, the completed predictor optimization report branch.
- Data: `/mnt/localnvme/glm53_predictor_architecture_cpu_20261007`.
- Environment: existing CPU evaluation virtualenv, Python 3.12, scikit-learn 1.9.1,
  NumPy 2.5.3. CUDA is hidden; training and inference use one CPU thread per job.
- Nine model/architecture combinations, 16 grouped outer tasks. The comparison
  includes wide-base and matched-data controls, episode and workload holdouts,
  signed residual training, and exported inference equivalence checks.

## Audit correction before accepting results

The first four pilot fits were stopped before any outer result was completed.
Inspection found that the duration floor used in the selection objective was
being applied to signed correction outputs. This can alter configuration
selection incorrectly. Selection now reconstructs the complete duration by
adding the validation base prediction before applying the serving floor.

A regression check with a negative correction distinguishes the correct choice
from the floored-residual choice. All three CPU contract checks pass. The full
16-task run was restarted; terminated pilot fits contribute no reported results.
The original pilot logs are retained in the artifact directory.

## Active run

The full command in `README.md` runs four CPU fitting workers and then the
single-process frozen inference benchmark. Per-fold `completed.json` files are
the acceptance markers; `split_audit.json`, `cross_fit_audit.json`, `tuning.json`,
and `predictions.npz` record validation and selection details. All accepted
exports must match the fitted predictions within 0.0001 ms.

The report rechecks coverage, episode exclusion, finite predictions, paired
metrics, and export agreement before generating summary tables and plots.
