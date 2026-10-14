import json,time
from pathlib import Path
from experiments.load_only_overlap_20261011 import workloads
root=Path('/mnt/localnvme/jeongyoon/load_only_overlap_20261011/datasets')
manifest=[]
started=time.monotonic()
for seed,count in [(12500,8),(12501,64),(12502,64)]:
 for spec in workloads.homogeneous_grid():
  path=root/f"{spec['id']}_s{seed}_r{count}.json.gz"
  workloads.build(path,spec,count,seed)
  manifest.append(dict(path=str(path),sha256=workloads.digest(path),seed=seed,requests=count,spec=spec))
  print(spec['id'],seed,count,len(manifest),round(time.monotonic()-started,1),flush=True)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
