"""Cross-check cumulative-prefix reuse against an independent pairwise oracle."""
import json,tempfile,math
from pathlib import Path
from analyze import hashed_trace
ROOT=Path(__file__).resolve().parent
checks=[]
manifest=json.loads((ROOT/'download_manifest.json').read_text())
with tempfile.TemporaryDirectory(dir=ROOT,prefix='validation_') as tmp:
 for item in manifest['files']:
  if item['dataset']=='tracelab':continue
  block=512 if item['dataset']=='mooncake' else 16
  with Path(item['path']).open() as f:rows=[json.loads(next(f)) for _ in range(150)]
  rows.sort(key=lambda r:r['timestamp'])
  path=Path(tmp)/item['name'];path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
  got=hashed_trace(dict(dataset=item['dataset'],path=str(path)))['first_200_reusable_prefix_tokens']
  prior=[];expected=[]
  for row in rows:
   cur=row['hash_ids'][:row['input_length']//block];best=0
   for old in prior:
    common=0
    for a,b in zip(cur,old):
     if a!=b:break
     common+=1
    best=max(best,common)
   expected.append(best*block);prior.append(cur)
  assert got==expected,item['name']
  checks.append(dict(file=item['name'],requests=150,pairwise_oracle_matches=True))
 # Equal content blocks following different prefixes cannot reuse KV.
 path=Path(tmp)/'conditional.jsonl';rows=[{'timestamp':i,'input_length':48,'output_length':1,'hash_ids':h} for i,h in enumerate([[1,7,8],[2,7,8],[1,7,9]])]
 path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 assert hashed_trace(dict(dataset='bailian',path=str(path)))['first_200_reusable_prefix_tokens']==[0,0,32]
(ROOT/'validation.json').write_text(json.dumps({'prefix_oracle':checks,'shared_nonprefix_blocks_do_not_count':True},indent=2)+'\n')
print('PASS:150requests from each7hash traces match pairwise longest-prefix oracle; independent block matches rejected')
