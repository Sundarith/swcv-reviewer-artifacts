"""Native A-RAG/RAGentA adapters; deferred GPU search waves and exact request replay."""
import sys,json,time,hashlib,threading,csv,importlib.util,traceback,gzip,shutil,os
from pathlib import Path
from types import SimpleNamespace as NS
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests,tiktoken
from transformers import AutoTokenizer,TokenizersBackend
R=Path('__EXPERIMENT_WORKSPACE__');sys.path[:0]=['__LOCAL_ASSET_ROOT__/agentic-graphrag-benchmarks/src','__LOCAL_ASSET_ROOT__/agentic-graphrag-benchmarks/baselines/A-RAG/src']
from agentic_graphrag.benchmark.schema import BenchmarkExample
from agentic_graphrag.benchmark.retrieval import RetrievalHit
profile_path=Path(sys.argv[1]);method=sys.argv[2];O=Path(sys.argv[3]);O.mkdir(parents=True,exist_ok=True);cfg=json.loads(profile_path.read_text());loader=TokenizersBackend if cfg.get('tokenizer_loader')=='TokenizersBackend' else AutoTokenizer;tokenizer=loader.from_pretrained(cfg['tokenizer_path'],local_files_only=True)
D=R/'logs/experiments/rcm335_nvd_first_fixed_pool_all_models_20261018_v1/shared';prefix=json.loads((D.parent/'encoding_config.json').read_text())['query_prefix'];allowed=set(json.loads((R/'analysis/ctibench1000_final_group_filters_20261107/source_supported_exclusions.json').read_text())['filtered_eligible_rows']);docs={i:json.loads(l) for i,l in enumerate((D/'documents.jsonl').open()) if i in allowed};chunkmap={d['id']:d['text'] for d in docs.values()};queries=[json.loads(l) for l in (R/'logs/experiments/rcm1000_source_priority_all_models_20261017_v1/shared/queries.jsonl').open()]
cache=json.loads((O/'retrieval_cache.json').read_text()) if (O/'retrieval_cache.json').exists() else {};pending=set();lock=threading.Lock();local=threading.local();encoder=tiktoken.encoding_for_model('gpt-4o')
class Deferred(BaseException):pass
class StorageUnavailable(BaseException):pass
def check_storage():
 if shutil.disk_usage(O).free<5*1024**3:raise StorageUnavailable('Below5GiB free; stop before new API calls')
def append(path,r):
 data=(json.dumps(r,default=str)+'\n').encode();data=gzip.compress(data) if str(path).endswith('.gz') else data
 with lock,path.open('ab') as stream:
  stream.write(data);stream.flush();os.fsync(stream.fileno())
class Embed:
 def embed(self,texts):return NS(embeddings=texts,prompt_tokens=0,latency_seconds=0)
class Index:
 query_instruction=prefix
 def search(self,text,*,top_k):
  if text not in cache:
   with lock:pending.add(text)
   raise Deferred()
  assert top_k<=100
  return tuple(RetrievalHit(chunk_id=docs[j]['id'],text=docs[j]['text'],score=score) for j,score in cache[text][:top_k])
class Chat:
 def request(self,payload):
  index=local.call_index;local.call_index+=1;path=O/f'requests_{local.qid}.jsonl.gz';saved=local.saved
  if index<len(saved):
   record=saved[index];assert record['request']==payload,('Replay request changed',local.qid,index)
   if record.get('error'):raise RuntimeError(record['error'])
   return record['response']
  ids=tokenizer.apply_chat_template(payload['messages'],tools=payload.get('tools'),tokenize=True,return_dict=False,add_generation_prompt=True,chat_template=Path(cfg['chat_template']).read_text(),**cfg['template']);assert len(ids)+payload['max_tokens']<=cfg['context'],('Context limit',local.qid,index,len(ids))
  for attempt in range(1,5):
   try:
    check_storage()
    response=requests.post('http://127.0.0.1:8480/v1/chat/completions',json=payload,timeout=1800);response.raise_for_status()
   except requests.RequestException as e:
    append(O/'request_attempts.jsonl',{'qid':local.qid,'index':index,'attempt':attempt,'error':str(e)})
    if attempt<4:time.sleep([1,2,4][attempt-1]);continue
    append(path,{'request':payload,'error':str(e)});raise
   try:body=response.json();assert body['choices']
   except Exception:
    append(path,{'request':payload,'error':'Malformed returned response; no retry','raw':response.text});raise
   append(path,{'request':payload,'response':body,'attempts':attempt});return body
 def generate(self,messages,*,max_tokens,temperature=0,seed=None,stop=None,tools=None,tool_choice=None,**other):
  assert not other,other;payload={'model':cfg['model_name'],'messages':messages,'max_tokens':min(max_tokens,512),'temperature':0,'n':1}
  if stop:payload['stop']=list(stop)
  if tools:payload['tools']=tools;payload['tool_choice']=tool_choice or 'auto'
  started=time.monotonic();body=self.request(payload);c=body['choices'][0];u=body.get('usage') or {}
  return NS(content=c['message'].get('content') or '',tool_calls=tuple(c['message'].get('tool_calls') or []),finish_reason=c.get('finish_reason'),usage=NS(prompt_tokens=u.get('prompt_tokens',0),completion_tokens=u.get('completion_tokens',0),total_tokens=u.get('total_tokens',0)),endpoint='http://127.0.0.1:8480/v1',response_id=body.get('id'),model=body.get('model'),latency_seconds=time.monotonic()-started)
chat=Chat()
def sparse(query,k):
 response=requests.post('http://127.0.0.1:8497/search',json={'query':query,'k':k},timeout=120);response.raise_for_status();return [(x['document']['id'],x['document']['text'],x['score']) for x in response.json()]
if method=='ragenta':
 import agentic_graphrag.benchmark.ragenta_diagnostic as native
 # Probe original label log probabilities without adding a logit bias or output schema.
 def probabilities(self,prompt,*,role):
  payload={'model':cfg['model_name'],'messages':[{'role':'user','content':prompt}],'max_tokens':2,'temperature':0,'n':1,'logprobs':True,'top_logprobs':20}
  body=chat.request(payload);items=body['choices'][0]['logprobs']['content'][0]['top_logprobs'];by={x['token'].strip():x['logprob'] for x in items}
  assert 'Yes' in by and 'No' in by,'Native judge labels absent from unconstrained top-logprobs'
  self.calls.append({'kind':'pairwise_logprob','role':role,'request':payload,'response':body,'scoring_contract':'actual unconstrained Yes/No token log probabilities, no logit bias'})
  return {'Yes':by['Yes'],'No':by['No']}
 native._ControlledAgent.get_log_probs=probabilities
 def system():
  retriever=native.RAGentAHybridRetriever(Index(),Embed(),query_instruction=prefix,elasticsearch_endpoint='http://127.0.0.1:8497',elasticsearch_index='filtered',alpha=.65,top_k=20,sparse_search=sparse)
  return native.RAGentADiagnosticSystem(retriever,logprob_endpoint='http://127.0.0.1:8480/v1/chat/completions',model=cfg['model_name'],top_k=20,max_followups=1,generation_max_tokens=512)
else:
 from arag.core.context import AgentContext
 from arag.tools.read_chunk import ReadChunkTool
 from arag.tools.registry import ToolRegistry
 from agentic_graphrag.benchmark.arag_naive_diagnostic import AragNaiveDiagnosticSystem,NaiveEmbeddingSearchTool,ChunkReadAliasTool
 from agentic_graphrag.benchmark.arag_system import AragAgentConfig
 prompt=(R/'logs/experiments/rcm335_phi_arag_custom_server2_full335_20261025_v2/system_prompt.txt').read_text()
 def count(messages):return len(tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,chat_template=Path(cfg['chat_template']).read_text(),**cfg['template']))
 def system():
  registry=ToolRegistry();registry.register(NaiveEmbeddingSearchTool(Index(),Embed(),query_instruction=prefix,count_tokens=lambda x:len(encoder.encode(x))))
  reader=ReadChunkTool.__new__(ReadChunkTool);reader.chunks_dict=chunkmap;reader.tokenizer=encoder;registry.register(ChunkReadAliasTool(reader))
  return AragNaiveDiagnosticSystem(registry,AgentContext,prompt,message_token_counter=count,config=AragAgentConfig(max_loops=15,max_token_budget=cfg['context']-512,max_tokens_per_call=512),source_commit='a44de6b')
completed={str(x['question_id']) for x in [json.loads(l) for l in (O/'predictions.jsonl').open()]} if (O/'predictions.jsonl').exists() else set()
(O/'protocol.json').write_text(json.dumps({'architecture':method,'corpus':120238,'workers':16,'per_generation_cap':512,'temperature':0,'deferred_search':'Replay exact cached requests; suspend all model calls before dualGPU query embedding; no repeated API answers','native_ARAG':'released dense search/chunk read loop,15turn maximum','native_RAGentA':'released role workflow,hybridalpha.65,top20,one followup; real label logprobs without logit bias','profile_sha256':hashlib.sha256(profile_path.read_bytes()).hexdigest(),'gold_opened':False},indent=2))
def one(q):
 local.qid=str(q['row_index']);local.call_index=0;path=O/f'requests_{local.qid}.jsonl.gz';local.saved=[json.loads(l) for l in gzip.open(path,'rt')] if path.exists() else []
 try:
  result=system().run(BenchmarkExample(example_id=local.qid,dataset='cti-rcm1000',question=q['prompt'],answers=(),passages=(),answerable=False),chat)
  answer=result.trace.get('raw_final_response',result.prediction);record={'question_id':local.qid,'content':answer,'trace':result.trace}
 except Deferred:return 'deferred'
 except Exception as e:record={'question_id':local.qid,'content':'','pipeline_failure':str(e),'traceback':traceback.format_exc()}
 append(O/'predictions.jsonl',record);return 'completed'
with ThreadPoolExecutor(max_workers=16) as pool:
 futures=[pool.submit(one,q) for q in queries if str(q['row_index']) not in completed]
 for count,f in enumerate(as_completed(futures),1):f.result();(O/'progress.json').write_text(json.dumps({'processed_this_wave':count,'previous_completed':len(completed),'total':1000}))
if pending:
 (O/'pending_queries.json').write_text(json.dumps(sorted(pending)));raise SystemExit(75)
records=[json.loads(l) for l in (O/'predictions.jsonl').open()];assert len(records)==len({x['question_id'] for x in records})==1000
path=O/'predictions.jsonl';(O/'prediction_freeze.json').write_text(json.dumps({'n':1000,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'gold_opened':False}))
module=R/'analysis/cticonnect_style_rcm335_20261105/vendor/metrics.py';spec=importlib.util.spec_from_file_location('final_metrics',module);metrics=importlib.util.module_from_spec(spec);sys.modules[spec.name]=metrics;spec.loader.exec_module(metrics)
with (R/'data/cti-bench/data/cti-rcm.tsv').open() as f:gold=list(csv.DictReader(f,delimiter='\t'))
items=[metrics.score_id_item(p['content'],{'target_type':'cwe','target_id':'CWE-'+str(int(gold[int(p['question_id'])]['GT'].split('-')[1]))},task='rcm',item_id=p['question_id']) for p in records]
(O/'report.json').write_text(json.dumps({'n':1000,'official_mean_question_metrics':metrics.aggregate_scores(items)['overall'],'pipeline_failures':sum('pipeline_failure' in r for r in records)},indent=2))
