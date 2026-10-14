import json,time
from pathlib import Path
from experiments.load_only_overlap_20261011 import workloads
root=Path('/mnt/localnvme/jeongyoon/load_only_overlap_20261011/datasets');manifest=[]
for seed in [12601,12602]:
 for size in [192,384]:
  path=root/f'evolving_pool{size}_s{seed}_r300.json.gz'
  data=workloads.evolving(path,300,seed,prefix_pool=size)
  assert data['seed']==seed and data['prefix_pool']==size and len(data['requests'])==300
  manifest.append(dict(path=str(path),sha256=workloads.digest(path),seed=seed,prefix_pool=size,requests=300))
  print(size,seed,len(manifest),flush=True)
(root/'evolving_manifest.json').write_text(json.dumps(manifest,indent=2))
