# Joint support for frozen shape calibration

Branch: `experiment/shape-joint-guard-20261010`.
Parent: `565dc1b`. Preserve all worker and sealing policies from the parent.

The original calibration checks separate request, context and suffix
bounds. Crossed warmups at 15552/3072 showed errors as large as 379 ms
inside those bounds. Warmups are excluded from serving selection; this
is evidence that marginal support is insufficient for the fitted model.

Schema 2 adds an explicit suffix/context support interval. Derive the
interval only from the original training rows, with a frozen 5% margin.
Keep every coefficient and original 90 ms fallback unchanged. Check
the conservative minimum-suffix/maximum-context and maximum-suffix/
minimum-context ratios, so untrained mixed compositions also fall back.
Schema 1 artifacts retain their prior behavior. This guard preserves
narrow training coverage; it does not make the calibration general.

Runtime tests cover supported interpolation, unsupported crossed
geometry inside marginal bounds, preservation of old artifacts, and
rejection of invalid guard bounds. Run before any serving comparison.
The ongoing `yc2` allocation keeps its original frozen checkpoint and
does not use this branch. Broader geometry fitting remains a separate
optimization and requires independent serving seeds.

```bash
PYTHONPATH="$PWD" .venv/bin/python \
  experiments/shape_joint_guard_20261010/build.py \
  /mnt/localnvme/jeongyoon/shape_calibration_20261010/frozen_fit.json \
  /mnt/localnvme/jeongyoon/shape_calibration_20261010/ns1_rows.json \
  /mnt/localnvme/jeongyoon/shape_joint_guard_20261010/frozen_guard.json
```
