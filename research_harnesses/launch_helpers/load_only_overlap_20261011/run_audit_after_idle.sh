#!/usr/bin/env bash
set -euo pipefail
cd /mnt/localnvme/jeongyoon/vllm_h2d_block_fast_20261011
for attempt in $(seq 1 240); do
  if ! kill -0 2964154 2>/dev/null; then break; fi
  sleep 5
done
if kill -0 2964154 2>/dev/null; then exit 70; fi
if test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"; then exit 71; fi
.venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_gpu_worker.py -q -k 'block_readiness or store_reuse or native_store or native_wait' > /mnt/localnvme/jeongyoon/load_only_overlap_20261011/block_fast_gpu_tests2.log 2>&1
.venv/bin/python -m experiments.load_only_overlap_20261011.serving \
 --output /mnt/localnvme/jeongyoon/hb11qa --allocation-seconds 3600 \
 --workloads short16k mixed_supported --repeats 1 --requests 32 --seed-start 12421 \
 --initial-barrier --dma-audit \
 --arm fullsafe=/mnt/localnvme/jeongyoon/setup_prediction_20261011/runtime/frozen_A.json \
 --idle-arm fullsafe=1 --block-arm fullsafe > /mnt/localnvme/jeongyoon/load_only_overlap_20261011/block_audit_serving_a.log 2>&1
