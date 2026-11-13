"""Gold-free CTIConnect-specific overlap screening and cached-vector sensitivity."""
import collections
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

O=Path(__file__).resolve().parent
ASSETS=Path(os.environ.get('SWCV_ASSET_ROOT', str(O/'assets')))
P=ASSETS
C=ASSETS/'description_cache'
S=ASSETS
MODEL=Path(os.environ['SWCV_ENCODER_PATH'])
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.open()]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2)+'\n')
def state(s,**kw):save('job.json',dict(status=s,pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),at=time.time(),**kw))
def lev(a,b):
 if len(a)<len(b):a,b=b,a
 prior=list(range(len(b)+1))
 for i,ca in enumerate(a,1):
  current=[i]
  for j,cb in enumerate(b,1):current.append(min(current[-1]+1,prior[j]+1,prior[j-1]+(ca!=cb)))
  prior=current
 return prior[-1]
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 protocol=read(O/'overlap_protocol.json')
 for name,expected in protocol['external_assets_sha256'].items():assert sha(ASSETS/name)==expected,name
 for name,expected in protocol['encoder_hashes'].items():assert sha(MODEL/name)==expected,name
 for name,expected in protocol['local_sha256'].items():assert sha(O/name)==expected,name
 source=rows(O/'source_descriptions.jsonl');cache_rows=rows(C/'description_inputs.jsonl')
 cache=[np.load(C/f'vectors_gpu{g}.npy',mmap_mode='r') for g in (0,1)]
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side='left')
 missing=[i for i,s in enumerate(source) if s['description_cache_index'] is None]
 lengths=[len(tokenizer(source[i]['description'],truncation=False)['input_ids']) for i in missing]
 assert all(n<=8192 for n in lengths)
 save('embedding_preflight.json',{'sources':len(source),'cached':len(source)-len(missing),'new':len(missing),'max_new_tokens':max(lengths,default=0),'no_truncation':True})
 source_vectors=np.empty((len(source),2560),dtype=np.float32)
 for i,s in enumerate(source):
  j=s['description_cache_index']
  if j is not None:source_vectors[i]=cache[j%2][j//2]
 state('encoding',new=len(missing))
 def encode(gpu):
  torch.cuda.set_device(gpu)
  tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side='left')
  model=AutoModel.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(f'cuda:{gpu}').eval()
  assigned=missing[gpu::2]
  def worker(w):
   for i in assigned[w::8]:
    tokens=tok(source[i]['description'],return_tensors='pt',truncation=False)
    assert tokens['input_ids'].shape[1]<=8192
    tokens={k:v.to(f'cuda:{gpu}') for k,v in tokens.items()}
    with torch.inference_mode():
     hidden=model(**tokens).last_hidden_state[:,-1,:].float()
     source_vectors[i]=torch.nn.functional.normalize(hidden,dim=1).cpu().numpy()[0]
  with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
  del model;gc.collect();torch.cuda.empty_cache()
  return {'gpu':gpu,'new_sources':len(assigned),'workers':8}
 with ThreadPoolExecutor(max_workers=2) as pool:usage=list(pool.map(encode,(0,1)))
 np.save(O/'source_description_vectors.npy',source_vectors)
 save('embedding_receipt.json',{'usage':usage,'vectors_sha256':sha(O/'source_description_vectors.npy'),'input_sha256':sha(O/'source_descriptions.jsonl')})
 allowed=np.array(read(O.parent/'exclusions.json')['filtered_eligible_rows'],dtype=np.int64)
 docs=rows(P/'documents.jsonl');meta=[json.loads(d['raw_data']) for d in docs]
 reference_map={r['row']:i for i,r in enumerate(cache_rows) if r['kind']=='reference'}
 assert all(int(row) in reference_map for row in allowed)
 desc=[cache_rows[reference_map[int(row)]]['description'] for row in allowed]
 vectors=np.stack([cache[j%2][j//2] for j in [reference_map[int(row)] for row in allowed]])
 assert np.isfinite(vectors).all() and np.isfinite(source_vectors).all()
 torch.backends.cuda.matmul.allow_tf32=False
 state('screening')
 def screen(gpu):
  torch.cuda.set_device(gpu)
  matrix=torch.nn.functional.normalize(torch.tensor(vectors,device=f'cuda:{gpu}'),dim=1)
  pairs=[]
  for i in range(gpu,len(source),2):
   q=torch.nn.functional.normalize(torch.tensor(source_vectors[i],device=f'cuda:{gpu}'),dim=0)
   scores=(matrix@q).cpu().numpy()
   for j in np.flatnonzero(scores>=.95):pairs.append({'source_cve_id':source[i]['source_cve_id'],'question_ids':source[i]['question_ids'],'reference_cve_id':meta[int(allowed[j])]['identifier'],'reference_row':int(allowed[j]),'cosine':float(scores[j])})
  return pairs
 with ThreadPoolExecutor(max_workers=2) as pool:semantic=[p for part in pool.map(screen,(0,1)) for p in part]
 (O/'semantic_pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in sorted(semantic,key=lambda p:(p['source_cve_id'],p['reference_row']))))
 def grams(text):
  words=text.split();return {tuple(words[i:i+5]) for i in range(len(words)-4)}
 sg=[grams(s['description']) for s in source];inverted=collections.defaultdict(list)
 for i,gs in enumerate(sg):
  for g in gs:inverted[g].append(i)
 exact={}
 for i,s in enumerate(source):exact.setdefault(s['description'],[]).append(i)
 lexical=[]
 for j,text in enumerate(desc):
  gs=grams(text);counts=collections.Counter(i for gram in gs for i in inverted.get(gram,[]))
  candidates={i for i,intersection in counts.items() if intersection/(len(gs)+len(sg[i])-intersection)>.8}
  candidates.update(exact.get(text,[]))
  for i in candidates:
   t=source[i]['description'];similarity=1-lev(t,text)/max(len(t),len(text)) if t!=text else 1.
   if similarity>.8:
    intersection=len(gs&sg[i]);jac=intersection/len(gs|sg[i]) if gs|sg[i] else float(t==text)
    lexical.append({'source_cve_id':source[i]['source_cve_id'],'question_ids':source[i]['question_ids'],'reference_cve_id':meta[int(allowed[j])]['identifier'],'reference_row':int(allowed[j]),'jaccard':jac,'edit_similarity':similarity,'exact':t==text})
 (O/'lexical_pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in lexical))
 urlutil=module('source_url_rules',O/'source_rules.py')
 classifier=module('source_classifier',O/'source_rules.py')
 sourcekeys=collections.defaultdict(list)
 for s in source:
  for url in s['urls']:
   key=urlutil.urlkey(url)
   if key and classifier.classify(key)[0]:sourcekeys[key].append(s['source_cve_id'])
 provenance=read(ASSETS/'reference_provenance.json')
 shared=[]
 for row in allowed:
  sid=meta[int(row)]['identifier']
  for url in provenance.get(sid,{}).get('urls',[]):
   key=urlutil.urlkey(url)
   if key in sourcekeys:shared.append({'reference_row':int(row),'reference_cve_id':sid,'source_cve_ids':sorted(set(sourcekeys[key])),'canonical_url':key,'classification':classifier.classify(key)[1]})
 (O/'shared_disclosure_pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in shared))
 description_removed={p['reference_row'] for p in semantic+lexical}
 combined_removed=description_removed|{p['reference_row'] for p in shared}
 masks={'primary':set(allowed.tolist()),'description_union':set(allowed.tolist())-description_removed,'description_plus_shared_disclosure':set(allowed.tolist())-combined_removed}
 for name,eligible in masks.items():
  removed=set(allowed.tolist())-eligible
  save(name+'_mask.json',{'condition':name,'removed_reference_rows':sorted(removed),'removed_reference_cve_ids':sorted(meta[i]['identifier'] for i in removed),'filtered_eligible_rows':sorted(eligible),'remaining_references':len(eligible),'primary_exclusion_sha256':sha(O.parent/'exclusions.json')})
 save('screening_summary.json',{'source_CVEs':len(source),'questions':290,'source_metadata_available':sum(s['source_metadata_available'] for s in source),'reference_descriptions':len(desc),'semantic_pairs':len(semantic),'semantic_references':len({p['reference_row'] for p in semantic}),'lexical_pairs':len(lexical),'lexical_references':len({p['reference_row'] for p in lexical}),'exact_pairs':sum(p['exact'] for p in lexical),'description_union_references':len(description_removed),'shared_disclosure_pairs':len(shared),'combined_references':len(combined_removed),'no_scoring_done':True})
 del vectors;gc.collect()
 state('retrieving')
 full_vectors=np.load(P/'document_embeddings.npy',mmap_mode='r')
 qvectors=np.load(ASSETS/'query_embeddings.npy',mmap_mode='r')
 queries=rows(ASSETS/'queries.jsonl');old={r['question_id']:r['hits'] for r in rows(ASSETS/'dense_ranked.jsonl')}
 def retrieve(gpu):
  torch.cuda.set_device(gpu);matrix=torch.tensor(np.asarray(full_vectors),device=f'cuda:{gpu}',dtype=torch.float32);out=[]
  for i in range(gpu,len(queries),2):
   scores=(matrix@torch.tensor(np.asarray(qvectors[i]),device=f'cuda:{gpu}')).cpu().numpy();qid=queries[i]['question_id'];variants={}
   for name,eligible in masks.items():
    permitted=np.array(sorted(eligible),dtype=np.int64);top=permitted[np.argsort(-scores[permitted],kind='stable')[:20]]
    hits=[{'row':int(row),'reference_cve_id':meta[int(row)]['identifier'],'similarity':float(scores[row]),'cwe_ids':meta[int(row)]['cwe_ids']} for row in top]
    if name=='primary':
     assert [h['row'] for h in hits]==[h['row'] for h in old[qid]],qid
     assert all(h['similarity']==prior['similarity'] for h,prior in zip(hits,old[qid])),qid
    votes=collections.Counter()
    for h in hits[:10]:
     for c in h['cwe_ids']:votes[c]+=h['similarity']
    variants[name]={'prediction':min(votes,key=lambda c:(-votes[c],c)),'neighbors':hits}
   out.append((i,{'question_id':qid,'conditions':variants}))
  return out
 with ThreadPoolExecutor(max_workers=2) as pool:predictions=sorted(p for part in pool.map(retrieve,(0,1)) for p in part)
 (O/'overlap_predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for i,row in predictions))
 save('overlap_prediction_freeze.json',{'sha256':sha(O/'overlap_predictions.jsonl'),'n':290,'k':10,'gold_loaded':False,'both_GPUs':True,'primary_top20_exact_rows_and_scores_match':True,'mask_sha256':{n:sha(O/(n+'_mask.json')) for n in masks}})
 state('completed',n=290,scoring_done=False)
if __name__=='__main__':
 state('preflight')
 try:main()
 except Exception:
  state('failed',error=traceback.format_exc());raise
