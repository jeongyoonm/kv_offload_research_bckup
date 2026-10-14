"""Audit the closed accepted scope, preserving planned omissions explicitly."""
import json,subprocess,time
from pathlib import Path
C=Path(__file__).resolve().parent
S=Path('/mnt/localnvme/jeongyoon/vllm_trace_recovery_20261011')
p=C.parent/'dc11os/summary.json'
while not p.exists() or not json.loads(p.read_text()).get('stopped'):
    time.sleep(30)
state=json.loads(p.read_text())
while not (C/'archives/manifest.json').exists():
    time.sleep(5)
manifest=json.loads((C/'archives/manifest.json').read_text())
assert manifest['report_returncode']==0
flags=[] if state['complete'] else ['--allow-partial']
with (C/'measurement_finalization.log').open('w') as log:
    for file in ('validate_final_measurements.py','assemble_report.py'):
        subprocess.run([str(S/'.venv/bin/python'),str(C/file),*flags],cwd=S,stdout=log,stderr=subprocess.STDOUT,check=True)
(C/'finalization_status.json').write_text(json.dumps({'measurement_gates_passed':True,'planned_grid_complete':state['complete'],'scope':'Closed accepted measurements only; archive integrity, visual review, shutdown and final package still require verification.'},indent=2)+'\n')
