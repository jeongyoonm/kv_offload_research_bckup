#!/usr/bin/env bash
set -euo pipefail
cd /mnt/localnvme/jeongyoon/vllm_diverse_cache_20261011
.venv/bin/python -m experiments.load_only_overlap_20261011.evaluate \
 --output /mnt/localnvme/jeongyoon/dc11oi --server-kind optimized --suite grid \
 --requests 64 --warmup-requests 8 --repeats 1 --concurrencies 32 --seed-start 12501 \
 --idle-target 4 --seconds 7200 \
 > /mnt/localnvme/jeongyoon/load_only_overlap_20261011/diverse_optimized_grid_c.log 2>&1
.venv/bin/python -m experiments.load_only_overlap_20261011.select_classes \
 /mnt/localnvme/jeongyoon/dc11oi /mnt/localnvme/jeongyoon/load_only_overlap_20261011/profiled_classes.json \
 > /mnt/localnvme/jeongyoon/load_only_overlap_20261011/selected_classes.log 2>&1
