#!/usr/bin/env bash
set -euo pipefail
cd /mnt/localnvme/jeongyoon/vllm_h2d_block_fast_20261011
.venv/bin/python -m experiments.load_only_overlap_20261011.serving \
 --output /mnt/localnvme/jeongyoon/hb11pb --allocation-seconds 4800 \
 --workloads mixed_supported --repeats 2 --requests 300 --seed-start 12431 --initial-barrier \
 --arm worker_old=/mnt/localnvme/jeongyoon/setup_prediction_20261011/runtime/frozen_A.json \
 --arm worker_block=/mnt/localnvme/jeongyoon/setup_prediction_20261011/runtime/frozen_A.json \
 --arm full_old=/mnt/localnvme/jeongyoon/setup_prediction_20261011/runtime/frozen_A.json \
 --arm full_block=/mnt/localnvme/jeongyoon/setup_prediction_20261011/runtime/frozen_A.json \
 --arm full_idle1=/mnt/localnvme/jeongyoon/setup_prediction_20261011/runtime/frozen_A.json \
 --arm full_idle4=/mnt/localnvme/jeongyoon/setup_prediction_20261011/runtime/frozen_A.json \
 --worker-arm worker_old --worker-arm worker_block \
 --block-arm worker_block --block-arm full_block --block-arm full_idle1 --block-arm full_idle4 \
 --idle-arm full_idle1=1 --idle-arm full_idle4=4 \
 > /mnt/localnvme/jeongyoon/load_only_overlap_20261011/block_performance_b.log 2>&1
