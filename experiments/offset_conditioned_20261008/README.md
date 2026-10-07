# Conditioned comparison of offset-only prediction

Branch: `experiment/glm53-offset-conditioned-comparison-20261008`.
Artifacts: `/mnt/localnvme/glm53_offset_conditioned_20261008`.

The original frozen, rotating-order study is preserved under
`/mnt/localnvme/glm53_offset_predictor_20261008`. Its first two 60K cells after
mixed workloads show roughly four seconds of GPU gaps, while later same-shape
cells have much smaller gaps. The slow position moves between predictors.
Later repeats show slow phases in other positions, too. Workload position alone
does not explain the variation. Long load-only waits and bursty batch sizes
appear under different predictors; the physical cause is not established.

This follow-up conditions every measured cell identically: 32 requests of the
same workload and prompt seed, using the common original tree checkpoint.
That conditioning episode is excluded. Both CPU and GPU logical caches are
then reset and primed anew for the measured 75-request episode. Priming has
zero CPU restoration reads in the original comparison; measured requests
restore their prefixes from CPU, with no GPU prefix hits.

Three balanced orders use fresh seeds 3301–3303. Each predictor receives 225
measured requests. The models are copied from the original frozen study, with
unchanged export hashes; no fitting or selection uses these new labels.
GLM 5.3 TP8, private CPU mappings, async H2D, input preparation, deadline,
slots, polling and original model selector remain the same. Conditioning does
not modify the serving policy. A one-hour watchdog stops the owned server.

Launch after the original server stops and the GPUs are free:

```bash
PYTHONPATH="$PWD" .venv/bin/python experiments/offset_conditioned_20261008/run.py
```

Use the CPU evaluation environment from the original study for `analyze.py`.
Its audit helpers can also evaluate this artifact root. Directory names for
excluded conditioning episodes begin with `rconditioning`, so the measured
cell glob does not include them.

This estimates performance after the specified conditioning episode. It does
not establish why the original phase effect occurred, and both unconditioned
and conditioned results must be reported.
