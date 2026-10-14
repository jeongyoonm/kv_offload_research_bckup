import hashlib,json,time,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
items=[]
m=json.loads((ROOT/'mooncake_tree.json').read_text())
for row in m['tree']:
 if row['path'].startswith('FAST25-release/traces/') and row['path'].endswith('.jsonl'):
  items.append(dict(dataset='mooncake',name=Path(row['path']).name,url=f"https://raw.githubusercontent.com/kvcache-ai/Mooncake/{m['sha']}/{row['path']}",commit=m['sha'],bytes=row['size'],git_blob_sha=row['sha']))
b=json.loads((ROOT/'bailian_tree.json').read_text())
for row in b['tree']:
 if row['path'].endswith('.jsonl'):
  pointer=(ROOT/(row['path']+'.lfs')).read_text().splitlines()
  items.append(dict(dataset='bailian',name=row['path'],url=f"https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/{b['sha']}/{row['path']}",commit=b['sha'],bytes=int(pointer[2].split()[1]),sha256=pointer[1].split('sha256:')[1]))
r=json.loads((ROOT/'tracelab_release.json').read_text())
for row in r['assets']:
 if row['name'].endswith('.duckdb'):
  items.append(dict(dataset='tracelab',name=f"tracelab_{r['tag_name']}.duckdb",url=row['browser_download_url'],release=r['tag_name'],bytes=row['size'],sha256='a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744'))
def download(item):
 p=ROOT/item['dataset']/item['name'];p.parent.mkdir(exist_ok=True)
 h=hashlib.sha256();blob=hashlib.sha1(f"blob {item['bytes']}\0".encode())
 req=urllib.request.Request(item['url'],headers={'User-Agent':'trace-characterization'})
 with urllib.request.urlopen(req,timeout=60) as src,p.open('wb') as dst:
  while data:=src.read(1024**2):dst.write(data);h.update(data);blob.update(data)
 assert p.stat().st_size==item['bytes'],item
 if item.get('sha256'):assert h.hexdigest()==item['sha256'],item
 if item.get('git_blob_sha'):assert blob.hexdigest()==item['git_blob_sha'],item
 out=dict(item,path=str(p),downloaded_sha256=h.hexdigest(),verified=True)
 print(json.dumps({'downloaded':item['name'],'bytes':item['bytes'],'verified':True}),flush=True)
 return out
with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(download,items))
(ROOT/'download_manifest.json').write_text(json.dumps({'utc':time.time(),'files':results},indent=2)+'\n')
