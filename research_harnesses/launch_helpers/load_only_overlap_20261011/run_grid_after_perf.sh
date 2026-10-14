#!/usr/bin/env bash
set -euo pipefail
cd /mnt/localnvme/jeongyoon/vllm_diverse_cache_20261011
for attempt in $(seq 1 720); do
  if test -f /mnt/localnvme/jeongyoon/hb11pb/summary.json; then break; fi
  sleep 5
done
.venv/bin/python - <<'PY'
import json,statistics,time
from pathlib import Path
root=Path('/mnt/localnvme/jeongyoon/load_only_overlap_20261011')
summary=json.loads(Path('/mnt/localnvme/jeongyoon/hb11pb/summary.json').read_text())
assert summary['complete'] and summary['stopped'] and len(summary['results'])==12
assert all(r['accepted'] for r in summary['results'])
assert json.loads((root/'block_audit_summary.json').read_text())['byte_correctness_accepted']
results=summary['results'];means={arm:statistics.mean(r['throughput_rps'] for r in results if r['label']==arm) for arm in {r['label'] for r in results}}
selected=max(['full_idle1','full_idle4'],key=means.get)
assert means[selected]>=means['full_old'], 'Investigate regression before broader evaluation'
choice=dict(selection_time=time.time(),throughput_means=means,selected_full=selected,idle_target=1 if selected=='full_idle1' else 4,predictor='frozen_A',deadline_lead_us=50000,h2d_block_readiness=True,reason='Highest mean throughput of two pre-ablation matched mixed-workload repeats; freeze before broader evaluation.')
(root/'frozen_policy_choice.json').write_text(json.dumps(choice,indent=2));print(choice,flush=True)
PY
.venv/bin/python -m experiments.load_only_overlap_20261011.evaluate \
 --output /mnt/localnvme/jeongyoon/dc11bg --server-kind baseline --suite grid \
 --requests 64 --warmup-requests 8 --repeats 1 --concurrencies 32 --seed-start 12501 --seconds 5400 \
 > /mnt/localnvme/jeongyoon/load_only_overlap_20261011/diverse_baseline_grid_a.log 2>&1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('/mnt/localnvme/jeongyoon/dc11bg/summary.json');d=json.loads(p.read_text());assert d['complete'] and d['stopped'] and len(d['results'])==16 and all(r['accepted'] for r in d['results'])
PY
DC11_IDLE_TARGET=$(.venv/bin/python -c "import json; print(json.load(open('/mnt/localnvme/jeongyoon/load_only_overlap_20261011/frozen_policy_choice.json'))['idle_target'])")
.venv/bin/python -m experiments.load_only_overlap_20261011.evaluate \
 --output /mnt/localnvme/jeongyoon/dc11og --server-kind optimized --suite grid \
 --requests 64 --warmup-requests 8 --repeats 1 --concurrencies 32 --seed-start 12501 \
 --idle-target "$DC11_IDLE_TARGET" --seconds 7200 \
 > /mnt/localnvme/jeongyoon/load_only_overlap_20261011/diverse_optimized_grid_a.log 2>&1
.venv/bin/python -m experiments.load_only_overlap_20261011.select_classes \
 /mnt/localnvme/jeongyoon/dc11og /mnt/localnvme/jeongyoon/load_only_overlap_20261011/profiled_classes.json \
 > /mnt/localnvme/jeongyoon/load_only_overlap_20261011/selected_classes.log 2>&1
