#!/usr/bin/env bash
set -euo pipefail
cd /mnt/localnvme/jeongyoon/vllm_diverse_cache_20261011
.venv/bin/python -m experiments.load_only_overlap_20261011.evaluate \
 --output /mnt/localnvme/jeongyoon/dc11bh --server-kind baseline --suite grid \
 --requests 64 --warmup-requests 8 --repeats 1 --concurrencies 32 --seed-start 12501 --seconds 5400 \
 > /mnt/localnvme/jeongyoon/load_only_overlap_20261011/diverse_baseline_grid_b.log 2>&1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('/mnt/localnvme/jeongyoon/dc11bh/summary.json');d=json.loads(p.read_text());assert d['complete'] and d['stopped'] and len(d['results'])==16 and all(r['accepted'] for r in d['results'])
PY
DC11_IDLE_TARGET=$(.venv/bin/python -c "import json; print(json.load(open('/mnt/localnvme/jeongyoon/load_only_overlap_20261011/frozen_policy_choice.json'))['idle_target'])")
.venv/bin/python -m experiments.load_only_overlap_20261011.evaluate \
 --output /mnt/localnvme/jeongyoon/dc11oh --server-kind optimized --suite grid \
 --requests 64 --warmup-requests 8 --repeats 1 --concurrencies 32 --seed-start 12501 \
 --idle-target "$DC11_IDLE_TARGET" --seconds 7200 \
 > /mnt/localnvme/jeongyoon/load_only_overlap_20261011/diverse_optimized_grid_b.log 2>&1
.venv/bin/python -m experiments.load_only_overlap_20261011.select_classes \
 /mnt/localnvme/jeongyoon/dc11oh /mnt/localnvme/jeongyoon/load_only_overlap_20261011/profiled_classes.json \
 > /mnt/localnvme/jeongyoon/load_only_overlap_20261011/selected_classes.log 2>&1
