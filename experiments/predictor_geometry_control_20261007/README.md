# Controlled geometry coverage

Compare geometry exclusion with equal-sized episode-purged controls using the
expanded GLM 5.3 TP8 dataset. All runs are CPU-only; no final replay labels enter
this study. Forecast-row training counts are 762, 730, and 815 per fold.
Controls match each source/phase count, without label-based sampling.

A geometry key is the sorted request `(uncached tokens, cached context tokens)`
multiset plus dispatch load count/tokens. Strict training removes all test keys.
Three outer episode splits and three seeds produce eighteen cells. Nine original
architectures receive independent inner episode tuning. Four selected follow-ups
are evaluated using the same purged data. Source weights and normalization are
refitted for each training cohort.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/predictor_geometry_control_20261007/study.py --workers 18
/mnt/localnvme/glm53_predictor_overnight_20261006/eval-venv/bin/python \
  experiments/predictor_geometry_control_20261007/report.py
```

Artifacts: `/mnt/localnvme/glm53_geometry_control_20261007`.
See REPORT.md for accuracy and caveats. Removing familiar geometries changes
training coverage even after count matching. This stress test does not select
or tune on the reused final benchmark, and does not establish serving benefits.
