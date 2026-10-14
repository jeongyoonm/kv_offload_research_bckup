import datetime,json,os,signal,time
from pathlib import Path
root=Path('/mnt/localnvme/jeongyoon/load_only_overlap_20261011')
stop=datetime.datetime(2026,10,11,17,7,0,tzinfo=datetime.timezone.utc).timestamp()
while time.time()<stop:
 time.sleep(min(45,stop-time.time()))
paths=[Path('/mnt/localnvme/jeongyoon/hb11pb')]
paths.extend(Path('/mnt/localnvme/jeongyoon').glob('dc11*'))
owned=[]
for path in paths:
 file=path/'server/process.json'
 if not file.exists():continue
 pid=json.loads(file.read_text())['pid']
 try:
  env=Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
  expected=f'TMPDIR={path}/tmp'.encode()
  if expected not in env or os.getpgid(pid)!=pid:continue
  os.killpg(pid,signal.SIGTERM);owned.append(dict(pid=pid,path=str(path)))
 except (OSError,ProcessLookupError):continue
(root/'deadline_stop.json').write_text(json.dumps(dict(stopped_at=time.time(),owned=owned),indent=2))
if owned:time.sleep(45)
for item in owned:
 try:
  env=Path(f"/proc/{item['pid']}/environ").read_bytes().split(b'\0')
  if f"TMPDIR={item['path']}/tmp".encode() in env:
   os.killpg(item['pid'],signal.SIGKILL)
 except (OSError,ProcessLookupError):pass
