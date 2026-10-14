"""Characterize pinned public traces without executing a GPU model."""
import collections,json,time,gc
from pathlib import Path
import numpy as np
import duckdb
ROOT=Path(__file__).resolve().parent

def stats(values):
 a=np.asarray(values,dtype=np.float64)
 return dict(n=len(a),min=float(a.min()),mean=float(a.mean()),p50=float(np.quantile(a,.5)),p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),max=float(a.max()),sum=float(a.sum()))

def hashed_trace(item):
 p=Path(item['path']);block=512 if item['dataset']=='mooncake' else 16
 rows=[]
 with p.open() as f:
  for line in f:
   if line.strip():rows.append(json.loads(line))
 original_times=[r['timestamp'] for r in rows]
 nonmonotonic=sum(a>b for a,b in zip(original_times,original_times[1:]))
 rows.sort(key=lambda r:r['timestamp'])
 isl=[];osl=[];reuse=[];nodes={};last_use={};gaps=[];hash_mismatch=[];types=collections.Counter();parents=0;next_node=1
 for index,r in enumerate(rows):
  n,o=int(r['input_length']),int(r['output_length']);assert n>=0 and o>=0
  hashes=r['hash_ids'];assert isinstance(hashes,list)
  if len(hashes)!=(n+block-1)//block:hash_mismatch.append(dict(row=index,isl=n,hashes=len(hashes)))
  full=min(n//block,len(hashes));parent=0;hit_blocks=0;miss=False
  for h in hashes[:full]:
   key=(parent,h);node=nodes.get(key)
   if node is None:
    miss=True;node=next_node;next_node+=1;nodes[key]=node
   elif not miss:
    hit_blocks+=1
    if (hit_blocks==full) or (hit_blocks==1):
     gaps.append((r['timestamp']-last_use[node])/(1000 if block==512 else 1))
   parent=node;last_use[node]=r['timestamp']
  isl.append(n);osl.append(o);reuse.append(hit_blocks*block)
  types[str(r.get('type','unspecified'))]+=1
  parents+=int(r.get('parent_chat_id',-1)!=-1)
  if index and index%20000==0:print(p.name,'rows',index,'nodes',len(nodes),flush=True)
 a=np.asarray(isl,dtype=np.int64);o=np.asarray(osl,dtype=np.int64);h=np.asarray(reuse,dtype=np.int64)
 out=dict(dataset=item['dataset'],file=p.name,requests=len(rows),source=item,block_tokens=block,
  isl=stats(a),osl=stats(o),reusable_prefix=stats(h),unreused_prompt=stats(a-h),first_200_reusable_prefix_tokens=reuse[:200],
  prefix_reuse_token_fraction=float(h.sum()/a.sum()),mean_request_reuse_fraction=float(np.mean(np.divide(h,a,out=np.zeros(len(a)),where=a>0))),
  requests_with_reuse_fraction=float(np.mean(h>0)),unique_full_prefix_blocks=len(nodes),unique_full_prefix_tokens=len(nodes)*block,
  types=dict(types),has_parent_fraction=parents/len(rows),duration_seconds=(original_times[-1]-original_times[0])/(1000 if block==512 else 1),
  timestamp_span_seconds=(max(original_times)-min(original_times))/(1000 if block==512 else 1),nonmonotonic_timestamp_pairs=nonmonotonic,
  hash_length_mismatches=len(hash_mismatch),hash_length_mismatch_examples=hash_mismatch[:5],
  isl_gt_81920_fraction=float(np.mean(a>81920)),isl_plus_osl_gt_81920_fraction=float(np.mean(a+o>81920)),
  zero_isl=int(np.sum(a==0)),zero_osl=int(np.sum(o==0)),
  reused_prefix_and_unreused_le_4096_fraction=float(np.mean((h>0)&(a-h<=4096))),
  reuse_definition='Cold-start prefix trie, timestamp order, insert previous prompts immediately, infinite capacity, only full blocks at released hash granularity; no generated-output KV or execution completion model. Hash equality is conditioned on all preceding blocks, never an independent block-membership count.')
 if gaps:out['first_or_last_matched_block_reuse_gap_seconds']=stats(gaps)
 del rows,nodes,last_use;gc.collect()
 return out

def trace_lab(item):
 c=duckdb.connect(item['path'],read_only=True)
 c.execute('set threads=4')
 valid='input_tokens_total > 0 and prefix_tokens >= 0 and newly_append_tokens >= 0 and output_tokens >= 0 and input_tokens_total=prefix_tokens+newly_append_tokens'
 counts=c.execute('select provider,count(*),count(distinct session_id),count(distinct "user") from rounds group by provider').fetchall()
 result=dict(dataset='tracelab',source=item,providers=[],raw_rows=c.execute('select count(*) from rounds').fetchone()[0],tool_calls=c.execute('select count(*) from tool_calls').fetchone()[0],valid_definition=valid)
 for provider in [None]+[row[0] for row in counts]:
  where=valid if provider is None else valid+' and provider=?'
  params=[] if provider is None else [provider]
  rows=c.execute('select input_tokens_total,prefix_tokens,newly_append_tokens,output_tokens from rounds where '+where,params).fetchnumpy()
  a=rows['input_tokens_total'];h=rows['prefix_tokens'];s=rows['newly_append_tokens'];o=rows['output_tokens']
  data=dict(provider=provider or 'all',valid_rows=len(a),isl=stats(a),prefix=stats(h),uncached_input=stats(s),osl=stats(o),
   observed_prefix_token_fraction=float(h.sum()/a.sum()),mean_request_hit_fraction=float(np.mean(h/a)),requests_with_cache_reads_fraction=float(np.mean(h>0)),
   isl_gt_81920_fraction=float(np.mean(a>81920)),isl_plus_osl_gt_81920_fraction=float(np.mean(a+o>81920)),
   cached_prefix_and_uncached_le_4096_fraction=float(np.mean((h>0)&(s<=4096))))
  groups=c.execute('select first_input_event_type,count(*),sum(prefix_tokens)::double/nullif(sum(input_tokens_total),0) from rounds where '+where+' group by first_input_event_type',params).fetchall()
  data['trigger_hit_rates']=[dict(trigger=t,rows=n,token_hit_fraction=v) for t,n,v in groups]
  result['providers'].append(data)
 result['raw_provider_counts']=[dict(provider=p,rows=n,sessions=s,users=u) for p,n,s,u in counts]
 result['invalid_or_incomplete_token_rows']=result['raw_rows']-result['providers'][0]['valid_rows']
 result['null_or_invalid_details']=dict(zip(['null_isl','null_prefix','null_append','null_osl','zero_isl','bad_input_sum'],c.execute('select count(*) filter(where input_tokens_total is null),count(*) filter(where prefix_tokens is null),count(*) filter(where newly_append_tokens is null),count(*) filter(where output_tokens is null),count(*) filter(where input_tokens_total=0),count(*) filter(where input_tokens_total!=prefix_tokens+newly_append_tokens) from rounds').fetchone()))
 return result

def main():
 manifest=json.loads((ROOT/'download_manifest.json').read_text())
 outputs=[]
 for item in manifest['files']:
  print('analyzing',item['name'],flush=True)
  row=trace_lab(item) if item['dataset']=='tracelab' else hashed_trace(item)
  outputs.append(row)
  (ROOT/'statistics.json').write_text(json.dumps({'utc':time.time(),'results':outputs},indent=2)+'\n')
  print('finished',item['name'],flush=True)

if __name__=="__main__":
 main()
