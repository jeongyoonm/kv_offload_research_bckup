# Independent context and suffix calibration

Branch: `experiment/shape-geometry-calibration-20261010`.
Code and data stay under `/mnt/localnvme/jeongyoon`.
Parent: `d1c9438`, including the frozen five-feature shape calibration.

The first calibration saw only context/suffix pairs 15552/832 and
58368/3072. Its new 30720/2048 serving case showed that prediction MAE
and serving benefit can move differently. Independent marginal bounds
do not establish support for arbitrary context/suffix combinations.

First validate the unchanged frozen correction on 15552/3072 and
58368/832. Compare the original fixed 90 ms correction/100 ms lead with
the frozen shape correction at both 100 ms and 50 ms leads, using two
paired 75-request
episodes and new seeds 9601/2. These are development observations for
any subsequent geometry refit; fresh serving seeds must validate that
refit. No labels from these cells select the initial frozen checkpoint.

All worker settings, preparation, native stores, allocator avoidance,
private CPU cache, GPU budget, load-only policy, model and selector
remain as in the parent. No new server starts while another allocation
owns the GPUs. This branch initially changes only the workload harness.

For continuous 300-request confirmations, use prefix working sets of
300 (16K), 150 (32K), 75 (60K), 250 (short context/long suffix), and
75 (long context/short suffix). Their prefix footprint exceeds the
observed 2,160,960-token GPU capacity. Prefixes, 64-token priming tails
and distinct measured suffixes fit within 87,220 CPU blocks of 64
tokens each. Serving checks must independently verify exact CPU
restoration, zero GPU prefix hits, unchanged frozen state and outputs.

```bash
PYTHONPATH="$PWD" .venv/bin/python \
  experiments/shape_calibration_20261010/run.py \
  --output /mnt/localnvme/jeongyoon/yc2 \
  --fit /mnt/localnvme/jeongyoon/shape_calibration_20261010/frozen_fit.json \
  --workloads short_long_suffix long_short_suffix \
  --arms control shape100 shape50 --seed-start 9601
```

Retain all rejected cells and failed checks. Do not interpret the
idealized readiness windows as guaranteed counterfactual throughput.
