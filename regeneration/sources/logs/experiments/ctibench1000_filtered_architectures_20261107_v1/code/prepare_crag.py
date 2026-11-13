"""Filtered Local CRAG: frozen native evaluator, three-way correction, strip refinement."""
import json,sys,os,hashlib,importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,torch
from transformers import AutoTokenizer,T5ForSequenceClassification
R=Path('__EXPERIMENT_WORKSPACE__');O=Path(__file__).resolve().parents[1];OLD=R/'logs/experiments/rcm335_crag_local_llama_pilot_20261023_v1'
code=OLD/'code/contracts.py';spec=importlib.util.spec_from_file_location('crag_contract',code);contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
prepared=[json.loads(l) for l in (O/'naive_rag20_prepared.jsonl').open()];docs={}
for line in (R/'logs/experiments/rcm335_nvd_first_fixed_pool_all_models_20261018_v1/shared/documents.jsonl').open():
 d=json.loads(line);m=json.loads(d['raw_data']);docs[m['identifier']]=d
allowed=set(json.loads((R/'analysis/ctibench1000_final_group_filters_20261107/source_supported_exclusions.json').read_text())['reference_cve_ids'])
torch.backends.cuda.matmul.allow_tf32=False
models=[];tokenizers=[]
for g in [0,1]:
 tok=AutoTokenizer.from_pretrained(OLD/'evaluator',local_files_only=True);model,info=T5ForSequenceClassification.from_pretrained(OLD/'evaluator',local_files_only=True,torch_dtype=torch.float32,output_loading_info=True);assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys']);model=model.to(f'cuda:{g}').eval();model.config.use_cache=False;models.append(model);tokenizers.append(tok)
for g in [0,1]:torch.cuda.synchronize(g)
(O/'local_crag_protocol.json').write_text(json.dumps({'method':'Local CRAG','evaluator_checkpoint':str(OLD/'evaluator'),'thresholds':{'correct':.59,'incorrect':-.99,'strip':-.5},'dense_k':20,'local_BM25_alternative_k':20,'retained_strips':5,'evaluator_text':'same first600characters of records used by common reader; query/doc token windows cover evaluator input without silent clipping','corpus':120238,'workers_per_gpu':8,'gold_opened':False,'contract_sha256':hashlib.sha256(code.read_bytes()).hexdigest()},indent=2))
def worker(w):
 g=w%2;torch.cuda.set_device(g);stream=torch.cuda.Stream(device=g);tok=tokenizers[g];result=[]
 def score(question,text):
  qids=tok(question,add_special_tokens=False)['input_ids'];sep=tok(' [SEP] ',add_special_tokens=False)['input_ids'];dids=tok(text,add_special_tokens=False)['input_ids'];values=[]
  for a in range(0,len(qids),192):
   prefix=qids[a:a+256]+sep;budget=512-len(prefix)-1;assert budget>0
   for b in range(0,max(1,len(dids)),budget):
    ids=prefix+dids[b:b+budget]+[tok.eos_token_id]
    with torch.cuda.stream(stream),torch.inference_mode():
     value=float(models[g](input_ids=torch.tensor([ids],device=f'cuda:{g}'),attention_mask=torch.ones((1,len(ids)),device=f'cuda:{g}',dtype=torch.long)).logits.float().reshape(-1)[0].cpu())
    values.append(value)
   if a+256>=len(qids):break
  return max(values)
 for i in range(w,1000,16):
  q=prepared[i];dense=[r['cve_id'] for r in q['references']];response=requests.post('http://127.0.0.1:8497/search',json={'query':q['prompt'],'k':20},timeout=120);response.raise_for_status();bm=[json.loads(x['document']['raw_data'])['identifier'] for x in response.json()]
  assert not(set(dense+bm)&allowed)
  texts={cid:docs[cid]['text'][:600] for cid in set(dense+bm)};initial=[score(q['prompt'],texts[cid]) for cid in dense];action=contract.action(initial);pool=contract.evidence_pool(action,dense,bm);strips=[]
  for cid in pool:
   for j,text in enumerate(contract.strips(texts[cid])):strips.append({'cve_id':cid,'strip_index':j,'text':text,'score':score(q['prompt'],text)})
  chosen=contract.select_strips(strips);record={**q,'variable_references':True,'references':[{'cve_id':x['cve_id'],'similarity':x['score'],'text_override':x['text']} for x in chosen]}
  if not chosen:record['pipeline_failure']='No evidence strips passed fixed evaluator threshold'
  result.append(record)
  (O/f'crag_progress_worker{w}.json').write_text(json.dumps({'last_query':i,'action':action}))
 return result
with ThreadPoolExecutor(max_workers=16) as pool:output=sorted([x for part in pool.map(worker,range(16)) for x in part],key=lambda x:int(x['question_id']))
(O/'local_crag_prepared.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in output))
