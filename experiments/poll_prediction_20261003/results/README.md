# Local experiment evidence

Strategy branch: `experiment/glm53-seal-predicted-readiness-20261003`.

Implemented runtime commit: `05413afedc869dd31bdeb1adaf0f1f7d75e2d50e`. The
combined screen runs all strategies from the hybrid branch; the
prediction-to-readiness comparison uses its separate branch. Per-phase
provenance records the actual evaluated checkout.

All 78 registered replays and 11,960 measured requests passed the cache,
transfer, output, GPU ownership and nominal-work gates. The summary preserves
weaker results and all matched controls.

Complete raw telemetry, requests and server logs:
`/mnt/localnvme/glm53_poll_prediction_20261003`. This archive contains reports,
plots, manifests, correctness results, CPU checks and compressed process/driver
logs. Large raw telemetry and model/native files remain at the original paths.
`index.json` records archive SHA256 hashes.

See [findings](summary/FINDINGS.md), [decisions](summary/DECISIONS.md) and [full
tables](summary/report.html).
