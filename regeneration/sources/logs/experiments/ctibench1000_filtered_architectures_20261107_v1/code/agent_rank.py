"""Batch deferred agent search requests on both GPUs; no concurrent answer model."""
import json,sys,os,subprocess,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch
R=Path('__EXPERIMENT_WORKSPACE__');O=Path(sys.argv[1]);wave=int(sys.argv[2]);W=O/f'wave{wave}';W.mkdir(exist_ok=False)
texts=json.loads((O/'pending_queries.json').read_text());assert texts
(W/'description_inputs.jsonl').write_text(''.join(json.dumps({'description':s})+'\n' for s in texts))
(W/'embed.py').write_text((R/'analysis/near_duplicate_description_gpu_20261107/embed.py').read_text());children=[subprocess.Popen([sys.executable,str(W/'embed.py')],env={**os.environ,'DESCRIPTION_GPU':str(g),'HF_HUB_OFFLINE':'1'}) for g in [0,1]];codes=[p.wait() for p in children];assert codes==[0,0],codes
v=np.empty((len(texts),2560),dtype=np.float32)
for g in [0,1]:v[g::2]=np.load(W/f'vectors_gpu{g}.npy')
dv=np.load(R/'logs/experiments/rcm335_nvd_first_fixed_pool_all_models_20261018_v1/shared/document_embeddings.npy',mmap_mode='r');eligible=np.asarray(json.loads((R/'analysis/ctibench1000_final_group_filters_20261107/source_supported_exclusions.json').read_text())['filtered_eligible_rows'],dtype=np.int64)
torch.backends.cuda.matmul.allow_tf32=False;matrices=[torch.tensor(np.asarray(dv),dtype=torch.float32,device=f'cuda:{g}') for g in [0,1]]
for g in [0,1]:torch.cuda.synchronize(g)
def worker(w):
 g=w%2;torch.cuda.set_device(g);stream=torch.cuda.Stream(device=g);out=[]
 for i in range(w,len(texts),16):
  with torch.cuda.stream(stream):scores=(matrices[g]@torch.tensor(v[i],device=f'cuda:{g}',dtype=torch.float32)).cpu().numpy()
  near=eligible[np.argsort(-scores[eligible],kind='stable')[:100]];out.append((texts[i],[(int(j),float(scores[j])) for j in near]))
 return out
cache=json.loads((O/'retrieval_cache.json').read_text()) if (O/'retrieval_cache.json').exists() else {}
with ThreadPoolExecutor(max_workers=16) as pool:
 for part in pool.map(worker,range(16)):
  for text,hits in part:cache[text]=hits
(O/'retrieval_cache.json').write_text(json.dumps(cache));(W/'freeze.json').write_text(json.dumps({'queries':len(texts),'workers_per_gpu':8,'mask_records':120238,'gold_opened':False,'cache_sha256':hashlib.sha256((O/'retrieval_cache.json').read_bytes()).hexdigest()}))
