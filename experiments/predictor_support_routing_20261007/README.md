# Support routing

The geometry-exclusion control showed the envelope generalizing better than the
MLP to novel batch combinations. Test a training-support rule in a separate branch:
known geometry/dispatch-I/O signatures use the fixed blend or sparse MLP, and
unknown signatures use the suffix-guarded envelope. No gate thresholds or weights
are fitted. Bounds, support signatures and model parameters are immutable.

Signatures use eleven base features and three sparse geometry features. Runtime
ages, ongoing-load progress, queue lead and future timestamps do not select the
route. Aggregate signatures can alias arbitrary unseen geometries; the observed
corpus has no aliases. The model still consumes causal runtime features after
routing. This is not a latency lookup table.

Run with the existing CPU evaluation venv and one BLAS/OpenMP thread:

```sh
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/predictor_support_routing_20261007/route.py
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/predictor_support_routing_20261007/route.py --benchmark
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/predictor_support_routing_20261007/route.py --geometry
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/predictor_support_routing_20261007/publish.py
```

Freeze on episode macro MAE with the historical regression gate before evaluating
the reused final replay. The original blend remains eligible. No GPU throughput
or serving integration is part of this study. Artifacts reside under
`/mnt/localnvme/glm53_support_routing_20261007`.
