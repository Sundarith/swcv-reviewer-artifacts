import sys
sys.path.insert(0,'__EXPERIMENT_WORKSPACE__/logs/experiments/cticonnect_rcm290_full_suite_20261107_v1/code/full')
"""Generated-query GPU embeddings and ranking against the filtered corpus only."""
from common import CODE, OUT, QUERY_PATH, MASK, CONNECT, questions, score_native
import json,os,sys,subprocess,hashlib,re
from pathlib import Path
import numpy as np,torch
from concurrent.futures import ThreadPoolExecutor
R=Path('__EXPERIMENT_WORKSPACE__');M=Path(__file__).resolve().parents[1];D=R/'logs/experiments/rcm335_nvd_first_fixed_pool_all_models_20261018_v1/shared';Q=OUT / "inputs"
method=sys.argv[1];stage=Path(sys.argv[2]);source=stage/'outputs.jsonl';freeze=json.loads((stage/'prediction_freeze.json').read_text());assert hashlib.sha256(source.read_bytes()).hexdigest()==freeze['sha256']
records={int(x['row_index']):x for x in map(json.loads,source.read_text().splitlines())};queries=questions();docs=[json.loads(l) for l in (D/'documents.jsonl').open()];meta=[json.loads(d['raw_data']) for d in docs];cfg=json.loads((D.parent/'encoding_config.json').read_text());eligible=np.asarray(json.loads((MASK).read_text())['filtered_eligible_rows'],dtype=np.int64)
sys.path.insert(0,'__EXPERIMENT_WORKSPACE__/logs/experiments/rcm335_hyde_fusion_nonreasoning_full335_20261024_v1/native/fusion/code')
from contracts import rewrite_text
inputs=[];index={};failed={}
for i in range(290):
 r=records[i];text=r['prediction'].strip()
 if not text or r.get('request_error') or r.get('returned_response_error') or r.get('finish_reason')=='length':failed[i]='intermediate_failure';continue
 texts=[text] if method!='fusion' else (rewrite_text(text,r.get('finish_reason'),queries[i]['prompt']) or [])
 if method=='fusion' and not texts:failed[i]='no_accepted_rewrites';continue
 index[i]=list(range(len(inputs),len(inputs)+len(texts)))
 for t in texts:inputs.append({'description':('' if method=='hyde' else cfg['query_prefix'])+t})
if not inputs:
 allrows=[{'question_id':queries[i]['question_id'],'pipeline_failure':failed[i]} for i in range(290)]
 (stage/'retrieval.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in allrows))
 (stage/'retrieval_freeze.json').write_text(json.dumps({'n':290,'intermediate_failures':290,'gold_opened':False})+'\n')
 raise SystemExit(0)
(stage/'description_inputs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in inputs))
# Reuse the independently tested8-workers/GPU encoder, supplying exact prefix above.
encoder=(R/'analysis/near_duplicate_description_gpu_20261107/embed.py').read_text();(stage/'embed.py').write_text(encoder)
children=[]
for g in [0,1]:children.append(subprocess.Popen([sys.executable,str(stage/'embed.py')],env={**os.environ,'DESCRIPTION_GPU':str(g),'HF_HUB_OFFLINE':'1'}))
codes=[c.wait() for c in children];assert codes==[0,0],codes
vectors=np.empty((len(inputs),2560),dtype=np.float32)
for g in [0,1]:vectors[g::2]=np.load(stage/f'vectors_gpu{g}.npy')
dv=np.load(D/'document_embeddings.npy',mmap_mode='r');original=np.load(CONNECT/'query_embeddings.npy',mmap_mode='r');assert original.shape==(290,2560)
torch.backends.cuda.matmul.allow_tf32=False;matrices=[torch.tensor(np.asarray(dv),device=f'cuda:{g}',dtype=torch.float32) for g in [0,1]]
for g in [0,1]:torch.cuda.synchronize(g)
def worker(w):
 g=w%2;torch.cuda.set_device(g);stream=torch.cuda.Stream(device=g);result=[]
 for i in range(w,290,16):
  if i in failed:result.append({'question_id':queries[i]['question_id'],'pipeline_failure':failed[i]});continue
  ranked=[]
  with torch.cuda.stream(stream):
   if method=='fusion':
    sc=(matrices[g]@torch.tensor(original[i],device=f'cuda:{g}',dtype=torch.float32)).cpu().numpy();ranked.append(list(map(int,eligible[np.argsort(-sc[eligible],kind='stable')[:20]])))
   for ix in index[i]:
    sc=(matrices[g]@torch.tensor(vectors[ix],device=f'cuda:{g}',dtype=torch.float32)).cpu().numpy();ranked.append(list(map(int,eligible[np.argsort(-sc[eligible],kind='stable')[:20]])))
  if method=='fusion':
   totals={}
   for order in ranked:
    for rank,j in enumerate(order):totals[j]=totals.get(j,0)+1/(60+rank+1)
   near=sorted(totals,key=lambda j:(-totals[j],j))[:20];weights=totals
  else:near=ranked[0];weights={j:float(sc[j]) for j in near}
  q=queries[i];result.append({'question_id':queries[i]['question_id'],'row_index':i,'source_cve_id':q['source_cve_id'],'prompt':q['prompt'],'references':[{'cve_id':meta[j]['identifier'],'similarity':weights[j]} for j in near]})
 return result
with ThreadPoolExecutor(max_workers=16) as pool:allrows=sorted([x for part in pool.map(worker,range(16)) for x in part],key=lambda x:x['question_id'])
(stage/'retrieval.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in allrows));(stage/'retrieval_freeze.json').write_text(json.dumps({'method':method,'n':290,'intermediate_failures':len(failed),'k':20,'rrf_k':60 if method=='fusion' else None,'eligible':119889,'gold_opened':False,'retrieval_sha256':hashlib.sha256((stage/'retrieval.jsonl').read_bytes()).hexdigest()})+'\n')
