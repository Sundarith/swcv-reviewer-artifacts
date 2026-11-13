import json,hashlib,sys,re,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
R=Path('__EXPERIMENT_WORKSPACE__');M=Path(__file__).resolve().parents[1];D=R/'logs/experiments/rcm335_nvd_first_fixed_pool_all_models_20261018_v1/shared';Q=R/'logs/experiments/rcm1000_source_priority_all_models_20261017_v1/shared'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.open()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main(method):
 out=M/(method+'_prepared.jsonl');assert not out.exists()
 mask=R/'analysis/ctibench1000_final_group_filters_20261107/source_supported_exclusions.json';eligible=np.asarray(read(mask)['filtered_eligible_rows'],dtype=np.int64);assert len(eligible)==120238
 queries=rows(Q/'queries.jsonl');docs=rows(D/'documents.jsonl');meta=[json.loads(d['raw_data']) for d in docs]
 protocol={'method':method,'questions':1000,'eligible_records':120238,'top_k':20,'mask_sha256':sha(mask),'query_source_sha256':sha(Q/'queries.jsonl'),'documents_sha256':sha(D/'documents.jsonl'),'code_sha256':sha(Path(__file__)),'gold_or_predictions_used':False}
 if method=='bm25_rag20':
  from sklearn.feature_extraction.text import CountVectorizer
  v=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b\w\w+\b',dtype=np.float32)
  counts=v.fit_transform([docs[i]['text'] for i in eligible]);lengths=np.asarray(counts.sum(axis=1)).ravel();avg=lengths.mean();k1=1.5;b=.75
  freq=np.bincount(counts.indices,minlength=counts.shape[1]);idf=np.log1p((len(eligible)-freq+.5)/(freq+.5));normal=k1*(1-b+b*lengths/avg)
  counts.data=counts.data*(k1+1)/(counts.data+np.repeat(normal,np.diff(counts.indptr)))*idf[counts.indices];weights=counts.tocsc();analyze=v.build_analyzer();vocab=v.vocabulary_;selected=[]
  for q in queries:
   terms=sorted({vocab[t] for t in analyze(q['description']) if t in vocab});scores=np.asarray(weights[:,terms].sum(axis=1)).ravel() if terms else np.zeros(len(eligible))
   near=np.argsort(-scores,kind='stable')[:20];selected.append([(int(eligible[k]),float(scores[k])) for k in near])
  protocol.update(tokenizer='lowercase word tokens length>=2; no stopword list; query unique terms',BM25_k1=k1,BM25_b=b,IDF='log(1+(N-df+.5)/(df+.5))',text='unchanged full reference record; query description only')
 elif method=='reranked_rag20':
  import torch
  from transformers import AutoTokenizer,AutoModelForSequenceClassification
  torch.backends.cuda.matmul.allow_tf32=False
  qv=np.array(np.load(Q/'query_embeddings.npy'),copy=True);oldq=np.load(D/'query_embeddings.npy');subset=rows(D/'queries.jsonl')
  for k,q in enumerate(subset):qv[q['row_index']]=oldq[k]
  dv=np.load(D/'document_embeddings.npy',mmap_mode='r');matrices=[torch.tensor(np.asarray(dv),device=f'cuda:{g}',dtype=torch.float32) for g in [0,1]]
  for g in [0,1]:torch.cuda.synchronize(g)
  candidates={}
  def rank(w):
   g=w%2;torch.cuda.set_device(g);stream=torch.cuda.Stream(device=g);part=[]
   for i in range(w,1000,16):
    with torch.cuda.stream(stream):scores=(matrices[g]@torch.tensor(qv[i],device=f'cuda:{g}',dtype=torch.float32)).cpu().numpy()
    near=eligible[np.argsort(-scores[eligible],kind='stable')[:100]];part.append((i,list(map(int,near))))
   return part
  with ThreadPoolExecutor(max_workers=16) as pool:
   for part in pool.map(rank,range(16)):candidates.update(part)
  del matrices
  for g in [0,1]:
   with torch.cuda.device(g):torch.cuda.empty_cache()
  modelroot=Path('__LOCAL_ASSET_ROOT__/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3');ref=modelroot/'refs/main';revision=ref.read_text().strip() if ref.exists() else sorted((modelroot/'snapshots').iterdir())[-1].name;path=modelroot/'snapshots'/revision
  assets={str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()}
  protocol.update(reranker='BAAI/bge-reranker-v2-m3',revision=revision,model_asset_hashes=assets,candidate_k=100,reranker_input='512token query/document windows covering all input tokens; max relevance across windows; no dropped query/document tokens',workers_per_GPU=8,precision='FP16 reranker, FP32 outputs')
  (M/(method+'_protocol.json')).write_text(json.dumps(protocol,indent=2)+'\n')
  models=[];tokenizers=[]
  for g in [0,1]:
   tokenizers.append(AutoTokenizer.from_pretrained(path,local_files_only=True))
   check=tokenizers[g]('synthetic question','synthetic reference')['input_ids'];a=tokenizers[g]('synthetic question',add_special_tokens=False)['input_ids'];b=tokenizers[g]('synthetic reference',add_special_tokens=False)['input_ids'];assert check==[tokenizers[g].cls_token_id]+a+[tokenizers[g].sep_token_id]*2+b+[tokenizers[g].sep_token_id],'Unexpected pair template'
   models.append(AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16).to(f'cuda:{g}').eval())
  for g in [0,1]:torch.cuda.synchronize(g)
  def rerank(w):
   g=w%2;torch.cuda.set_device(g);stream=torch.cuda.Stream(device=g);result=[]
   for i in range(w,1000,16):
    scored=[]
    qids=tokenizers[g](queries[i]['description'],add_special_tokens=False)['input_ids']
    for j in candidates[i]:
     dids=tokenizers[g](docs[j]['text'],add_special_tokens=False)['input_ids'];values=[]
     for qa in range(0,len(qids),192):
      piece=qids[qa:qa+256];budget=512-len(piece)-tokenizers[g].num_special_tokens_to_add(pair=True);assert budget>0
      for da in range(0,max(1,len(dids)),budget):
       ids=[tokenizers[g].cls_token_id]+piece+[tokenizers[g].sep_token_id,tokenizers[g].sep_token_id]+dids[da:da+budget]+[tokenizers[g].sep_token_id]
       token={'input_ids':torch.tensor([ids]),'attention_mask':torch.ones((1,len(ids)),dtype=torch.long)};assert len(ids)<=512
       with torch.cuda.stream(stream),torch.inference_mode():
        token={k:v.to(f'cuda:{g}') for k,v in token.items()};value=float(models[g](**token).logits.float().reshape(-1)[0].cpu())
       assert np.isfinite(value);values.append(value)
      if qa+256>=len(qids):break
     assert values;scored.append((j,max(values)))
    result.append((i,sorted(scored,key=lambda x:(-x[1],x[0]))[:20]))
    if i%100==0:(M/f'rerank_progress_worker{w}.json').write_text(json.dumps({'last_query':i}))
   return result
  selected=[None]*1000
  with ThreadPoolExecutor(max_workers=16) as pool:
   for part in pool.map(rerank,range(16)):
    for i,near in part:selected[i]=near
 else:raise ValueError(method)
 records=[]
 for i,q in enumerate(queries):
  assert len(selected[i])==20
  records.append({'question_id':str(i),'row_index':i,'source_cve_id':q['cve_id'],'prompt':q['prompt'],'references':[{'cve_id':meta[j]['identifier'],'similarity':score} for j,score in selected[i]]})
 out.write_text(''.join(json.dumps(x)+'\n' for x in records));protocol['prepared_sha256']=sha(out);(M/(method+'_protocol.json')).write_text(json.dumps(protocol,indent=2)+'\n')
if __name__=='__main__':main(sys.argv[1])
