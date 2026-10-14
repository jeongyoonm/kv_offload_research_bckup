#!/usr/bin/env bash
set -euo pipefail
cd /mnt/localnvme/jeongyoon/vllm_h2d_block_fast_20261011
for attempt in $(seq 1 720); do
  if test -f /mnt/localnvme/jeongyoon/hb11qa/summary.json; then break; fi
  sleep 5
done
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('/mnt/localnvme/jeongyoon/hb11qa')
summary=json.loads((root/'summary.json').read_text())
assert summary['complete'] and summary['stopped']
assert all(r['accepted'] for r in summary['results'])
paths=list((root/'dma_audit').glob('rank*.jsonl'));assert len(paths)==8
ranks=[]
for path in sorted(paths):
 rows=[json.loads(l) for l in path.open()]
 assert all(r.get('mismatched_bytes',0)==0 and r.get('changed_cpu_blocks',0)==0 for r in rows),path
 count={d:sum(r['verified_bytes'] for r in rows if r['direction']==d) for d in ['H2D','D2H','before_forward']}
 assert all(v>0 for v in count.values()),(path,count)
 ranks.append(dict(rank_file=path.name,records=len(rows),verified_bytes=count,mismatched_bytes=0,changed_cpu_blocks=0))
Path('/mnt/localnvme/jeongyoon/load_only_overlap_20261011/block_audit_summary.json').write_text(json.dumps(dict(accepted=True,ranks=ranks),indent=2))
print('All-rank DMA and pre-forward audit passed',flush=True)
PY
.venv/bin/python -m experiments.load_only_overlap_20261011.serving \
 --output /mnt/localnvme/jeongyoon/hb11pa --allocation-seconds 4800 \
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
 > /mnt/localnvme/jeongyoon/load_only_overlap_20261011/block_performance_a.log 2>&1
