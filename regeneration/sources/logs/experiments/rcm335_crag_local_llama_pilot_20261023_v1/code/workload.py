"""Fixed one-attempt reader, real serving-token checks, row-index resumability."""
import csv,gzip,json,os,sys,time,threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import requests
from shared import P,S,rows,save,sha
from parser import catalog_answer_id
name=sys.argv[1];cfg=json.loads((P/'models.json').read_text())[name];out=P/'models'/name
allowed=set(json.loads((P/'allowed_ids.json').read_text()));entries=rows(S/'corrected.jsonl')
assert len(entries)==335 and len({r['row_index'] for r in entries})==335
lock=threading.Lock()
def append(path,value,compressed=False):
 with lock:
  with (gzip.open(path,'at') if compressed else path.open('a')) as f:f.write(json.dumps(value)+'\n')
def request(port,path,body):
 r=requests.post(f'http://127.0.0.1:{port}{path}',json=body,timeout=900)
 if r.status_code>=500 or r.status_code in (408,429):raise RuntimeError(f'Infrastructure HTTP{r.status_code}: {r.text[:800]}')
 r.raise_for_status();return r.json()
def payload(messages,budget=2048):return dict(model='Qwen/Qwen2.5-7B-Instruct',messages=messages,max_tokens=budget,temperature=.1,seed=0,top_p=1,n=1,chat_template_kwargs={'enable_thinking':True})
ports=[8001] if cfg['TP']==2 else [8001,8002]
if not (out/'preflight.json').exists():
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(cfg['snapshot'],local_files_only=True)
 audit=[]
 for entry in entries:
  body=payload(entry['messages']);count=request(8001,'/tokenize',dict(model=body['model'],messages=body['messages'],add_generation_prompt=True,chat_template_kwargs=body['chat_template_kwargs']))['count']
  local=len(tok.apply_chat_template(body['messages'],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True))
  assert count+2048<=cfg['context'],(entry['row_index'],count,cfg['context'])
  audit.append(dict(row_index=entry['row_index'],serving_input_tokens=count,local_input_tokens=local))
 import re
 for gpu in range(2):
  capacities=re.findall(r'GPU KV cache size: ([\d,]+) tokens',(out/f'answer_server_{gpu}.log').read_text());assert capacities
  assert max(x['serving_input_tokens']+2048 for x in audit)*8<=int(capacities[-1].replace(',','')),'Eight longest requests exceed measured KV capacity'
 def gate(i):
  port=ports[i%len(ports)];body=payload([{'role':'user','content':'Reply with only CWE-79.'}],32)
  result=request(port,'/v1/chat/completions',body)
  assert result['choices'][0]['finish_reason'] in ('stop','length')
  return dict(port=port,response=result)
 with ThreadPoolExecutor(max_workers=16) as pool:synthetic=list(pool.map(gate,range(16)))
 save(out/'preflight.json',dict(passed=True,questions=335,context=cfg['context'],output_budget=2048,token_audit=audit,concurrent_synthetic_results=synthetic))
# Validate count again per request so a model restart cannot bypass the serving contract.
def task(entry,gpu):
 port=8001 if cfg['TP']==2 else 8001+gpu;body=payload(entry['messages']);started=time.time()
 count=request(port,'/tokenize',dict(model=body['model'],messages=body['messages'],add_generation_prompt=True,chat_template_kwargs=body['chat_template_kwargs']))['count']
 assert count+2048<=cfg['context']
 try:result=request(port,'/v1/chat/completions',body)
 except Exception as exc:
  append(out/'infrastructure_requests.jsonl',dict(row_index=entry['row_index'],port=port,error=str(exc),at=time.time()))
  raise
 append(out/'raw_requests.jsonl.gz',dict(row_index=entry['row_index'],request=body,response=result,port=port,serving_input_tokens=count),True)
 choice=result['choices'][0];assert result['usage']['prompt_tokens']==count,(count,result['usage'])
 content=choice['message'].get('content') or '';pred=catalog_answer_id(content,choice['finish_reason'],allowed)
 # Reasoning, when separated by the serving parser, is retained in raw_response.
 return {k:entry[k] for k in ['row_index','cve_id','prompt']}|dict(prediction=pred,answer=content,finish_reason=choice['finish_reason'],usage=result['usage'],port=port,seconds=time.time()-started,error=None)
outputs=rows(out/'predictions.jsonl') if (out/'predictions.jsonl').exists() else [];done={r['row_index'] for r in outputs};assert len(done)==len(outputs)
remaining=[r for r in entries if r['row_index'] not in done]
# Eight request threads per independent GPU, sixteen total.
with ThreadPoolExecutor(max_workers=8) as a,ThreadPoolExecutor(max_workers=8) as b:
 futures=[(a if i%2==0 else b).submit(task,r,i%2) for i,r in enumerate(remaining)]
 for f in as_completed(futures):
  r=f.result();append(out/'predictions.jsonl',r);outputs.append(r)
  save(out/'progress.json',dict(done=len(outputs),total=335,at=time.time(),workers_total=16,workers_per_GPU=8 if cfg['TP']==1 else None,length=sum(r['finish_reason']=='length' for r in outputs),unresolved=sum(r['prediction'] is None for r in outputs)))
assert len(outputs)==335 and {r['row_index'] for r in outputs}=={r['row_index'] for r in entries}
# All immutable predictions are on disk BEFORE opening benchmark labels.
save(out/'predictions_frozen.json',dict(sha256=sha(out/'predictions.jsonl'),at=time.time(),n=335))
benchmark=list(csv.DictReader((P.parents[2]/'data/cti-bench/data/cti-rcm.tsv').open(),delimiter='\t'))
correct=0
for r in outputs:
 gold=benchmark[r['row_index']];assert gold['Prompt']==r['prompt'];correct+=r['prediction']==catalog_answer_id(gold['GT'],'stop',allowed)
usage=Counter()
for r in outputs:usage.update({k:r['usage'].get(k,0) for k in ['prompt_tokens','completion_tokens','total_tokens']})
result=dict(n=335,correct=correct,accuracy=correct/335,unresolved=sum(r['prediction'] is None for r in outputs),length=sum(r['finish_reason']=='length' for r in outputs),errors=0,usage=dict(usage),output_tokens=2048,retries=0,JSON_fallback=False,model=name,workers_total=16,scoring='Strict catalog CWE ID, no hierarchy credit',at=time.time())
save(out/'comparison.json',result);save(out/'completed.json',result)
print(json.dumps(result),flush=True)
