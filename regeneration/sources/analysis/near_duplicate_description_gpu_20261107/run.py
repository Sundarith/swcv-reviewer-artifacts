import os,json,time,hashlib,subprocess,traceback
from pathlib import Path
ROOT=Path('__EXPERIMENT_WORKSPACE__');OUT=Path(__file__).resolve().parent
P=ROOT/'logs/experiments/rcm335_nvd_first_fixed_pool_all_models_20261018_v1/shared'
Q=ROOT/'logs/experiments/rcm1000_source_priority_all_models_20261017_v1/shared'
E=ROOT/'logs/experiments/swcv_ctibench1000_exact_description_excluded_20261106_v1/eligible_rows.json'
HOST=ROOT/'scripts/local_llm_api_host.py';HS=ROOT/'logs/services/local_llm_api_20261106';PY='__LOCAL_ASSET_ROOT__/miniconda3/bin/python'
def read(p):return json.loads(p.read_text())
def save(n,d):
 p=OUT/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def ticks(pid):
 try:return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
 except FileNotFoundError:return None
def state(s,**kw):save('job.json',dict(status=s,pid=os.getpid(),start_ticks=ticks(os.getpid()),at=time.time(),**kw))
def compute():
 import numpy as np,torch
 from concurrent.futures import ThreadPoolExecutor
 torch.backends.cuda.matmul.allow_tf32=False
 eligible=np.asarray(read(E),dtype=np.int64)
 docs=[json.loads(l) for l in (P/'documents.jsonl').open()];qs=[json.loads(l) for l in (Q/'queries.jsonl').open()]
 dv=np.zeros((len(docs),2560),dtype=np.float32)
 allv=np.empty((len(eligible)+1000,2560),dtype=np.float32)
 for gpu in [0,1]:allv[gpu::2]=np.load(OUT/f'vectors_gpu{gpu}.npy')
 qv=allv[:1000];dv[eligible]=allv[1000:]
 assert np.isfinite(qv).all() and np.isfinite(dv).all()
 matrices=[]
 for gpu in [0,1]:
  torch.cuda.set_device(gpu)
  matrices.append(torch.nn.functional.normalize(torch.tensor(np.asarray(dv[eligible]),dtype=torch.float32,device=f'cuda:{gpu}'),dim=1))
 for gpu in [0,1]:torch.cuda.synchronize(gpu)
 def worker(worker_id):
  gpu=worker_id%2
  torch.cuda.set_device(gpu)
  stream=torch.cuda.Stream(device=gpu)
  stream.wait_stream(torch.cuda.default_stream(gpu))
  result=[]
  for i in range(worker_id,1000,16):
   with torch.cuda.stream(stream):
    query=torch.nn.functional.normalize(torch.tensor(qv[i],dtype=torch.float32,device=f'cuda:{gpu}'),dim=0)
    scores=(matrices[gpu]@query).cpu().numpy()
   for k in np.flatnonzero(scores>=.95):
    row=int(eligible[k]);result.append({'target_row':i,'target_cve_id':qs[i]['cve_id'],'reference_row':row,'reference_cve_id':json.loads(docs[row]['raw_data'])['identifier'],'cosine':float(scores[k])})
  return result
 with ThreadPoolExecutor(max_workers=16) as pool:parts=list(pool.map(worker,range(16)))
 pairs=sorted(sum(parts,[]),key=lambda x:(x['target_row'],x['reference_row']))
 (OUT/'matching_pairs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in pairs))
 report={}
 for threshold in [.95,.98]:
  selected=[x for x in pairs if x['cosine']>=threshold];rr=sorted({x['reference_row'] for x in selected});ids=sorted({x['reference_cve_id'] for x in selected})
  name=f'exclusions_{threshold:.2f}.json';save(name,{'threshold':threshold,'reference_rows':rr,'reference_cve_ids':ids,'filtered_eligible_rows':sorted(set(map(int,eligible))-set(rr)),'original_corpus_modified':False})
  report[str(threshold)]={'matching_pairs':len(selected),'unique_references_removed_in_variant':len(rr),'remaining_references':len(eligible)-len(rr),'affected_targets':len({x['target_row'] for x in selected}),'manifest_sha256':sha(OUT/name)}
 save('report.json',{'status':'complete','conditions':report,'gpu_query_counts':[500,500],'workers_per_gpu':8,'total_workers':16,'FP32':True,'TF32':False,'gold_or_predictions_loaded':False,'source_corpus_modified':False,'benchmark_scored':False})
if __name__=='__main__':
 if os.environ.get('SEMANTIC_COMPUTE')=='1':compute();raise SystemExit
 try:
  assert not (OUT/'report.json').exists(),'Preserve completed run'
  paths=[P/'documents.jsonl',P/'document_embeddings.npy',P/'queries.jsonl',P/'query_embeddings.npy',Q/'queries.jsonl',Q/'query_embeddings.npy',E,Path(__file__)]
  state('freezing_inputs')
  save('protocol.json',{'method':'SemDeDup-inspired benchmark-to-reference semantic sensitivity; exhaustive cosine instead of clustering','encoder':'new Qwen3-Embedding-4B description-only vectors; revision 5cf2132abc99cad020ac570b19d031efec650f2b','workers_per_gpu':8,'total_workers':16,'primary_threshold':.95,'stricter_threshold':.98,'threshold_selection':'fixed before scoring, not optimized on benchmark performance','representation':'description-only on both sides; same encoder, no task instructions, casefold/whitespace normalization, no metadata; all vectors newly encoded','normalization':'FP32 L2 normalize, TF32 disabled','comparison':'all1000 queries vs120671 eligible references;500 queries per GPU; no top-k truncation or transitive grouping','action':'separate filtered eligible-row manifests; preserve original corpus, benchmark questions, results','interpretation':'similarity-defined potential semantic overlap, not same-flaw or leakage-free proof','citation':'https://arxiv.org/abs/2303.09540','input_hashes':{str(p):sha(p) for p in paths}})
  state('waiting_for_existing_review')
  review=ROOT/'analysis/near_duplicate_full_review_20261106'
  while True:
   j=read(review/'job.json')
   if ticks(j['pid'])!=j.get('start_ticks'):break
   time.sleep(10)
  assert (review/'triage_freeze.json').exists(),'Review exited without frozen completion; preserve host and investigate'
  h=read(HS/'job.json');assert h['status']=='ready' and ticks(h['pid'])==h['start_ticks']
  profile=read(HS/'settings.json')['profile'];assert Path(profile).is_file()
  state('owned_API_handoff')
  subprocess.run([PY,str(HOST),'stop'],check=True)
  restore=True
  try:
   for _ in range(90):
    if not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():break
    time.sleep(1)
   else:raise RuntimeError('GPUs not free; no unowned process stopped')
   state('description_embedding')
   docs=[json.loads(l) for l in (P/'documents.jsonl').open()];qs=[json.loads(l) for l in (Q/'queries.jsonl').open()];eligible=read(E)
   norm=lambda text:' '.join(text.casefold().split())
   inputs=[{'kind':'target','row':i,'description':norm(q['description'])} for i,q in enumerate(qs)]
   inputs += [{'kind':'reference','row':i,'description':norm(docs[i]['text'].split('\n\n',1)[1].split('\n## ',1)[0])} for i in eligible]
   (OUT/'description_inputs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in inputs))
   model=Path('__LOCAL_ASSET_ROOT__/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b')
   save('encoding_freeze.json',{'input_sha256':sha(OUT/'description_inputs.jsonl'),'embed_code_sha256':sha(OUT/'embed.py'),'model_snapshot':str(model),'model_hashes':{str(x):sha(x) for x in model.rglob('*') if x.is_file()},'workers_per_gpu':8,'batch_per_worker':1,'max_tokens':8192,'overlength':'fail, never truncate','pooling':'last token; L2 normalized','dtype':'BF16 forward, FP32 vectors','same_encoding_both_sides':True})
   children=[subprocess.Popen(['__LOCAL_ASSET_ROOT__/miniconda3/envs/cyber-ft/bin/python',str(OUT/'embed.py')],env={**os.environ,'DESCRIPTION_GPU':str(gpu),'HF_HUB_OFFLINE':'1'}) for gpu in [0,1]]
   codes=[child.wait() for child in children]
   assert codes==[0,0],f'Embedding failure {codes}; no partial filtering'
   state('GPU_comparison')
   subprocess.run(['__LOCAL_ASSET_ROOT__/miniconda3/envs/cyber-ft/bin/python',str(Path(__file__).resolve())],env={**os.environ,'SEMANTIC_COMPUTE':'1'},check=True)
  finally:
   # Compute subprocess exits first, releasing all CUDA contexts before API restore.
   log=(OUT/'restored_host.log').open('ab')
   p=subprocess.Popen([PY,'-u',str(HOST),'serve'],cwd=ROOT,env={**os.environ,'CTI_HOST_CONFIG':profile,'HF_HUB_OFFLINE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   save('restored_host_owner.json',{'pid':p.pid,'profile':profile})
  state('completed')
 except Exception:
  state('failed',traceback=traceback.format_exc());raise
