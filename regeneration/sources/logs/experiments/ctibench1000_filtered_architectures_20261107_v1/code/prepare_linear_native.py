"""Rebuild native LinearRAG graph from filtered CVE-only passages; no old graph reuse."""
import os,sys,json,hashlib,subprocess,shutil
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from transformers import AutoTokenizer
R=Path('__EXPERIMENT_WORKSPACE__');M=Path(__file__).resolve().parents[1];O=Path('/dev/shm/swcv_linearrag_filtered_20261107');assert shutil.disk_usage('/dev/shm').free>25*1024**3,'Need25GiB scratch space for graph caches';O.mkdir(exist_ok=False);NATIVE=Path('__LOCAL_ASSET_ROOT__/agentic-graphrag-benchmarks/baselines/LinearRAG');sys.path.insert(0,str(NATIVE))
from src.utils import compute_mdhash_id
D=R/'logs/experiments/rcm335_nvd_first_fixed_pool_all_models_20261018_v1/shared';Q=R/'logs/experiments/rcm1000_source_priority_all_models_20261017_v1/shared'
def rows(p):return [json.loads(l) for l in p.open()]
def save(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
mask=json.loads((R/'analysis/ctibench1000_final_group_filters_20261107/source_supported_exclusions.json').read_text());allowed=set(mask['filtered_eligible_rows']);docs=rows(D/'documents.jsonl');queries=rows(Q/'queries.jsonl');cfg=json.loads((D.parent/'encoding_config.json').read_text());tokenizer=AutoTokenizer.from_pretrained(cfg['snapshot'],local_files_only=True);passages=[];source={}
for row in sorted(allowed):
 d=docs[row];ids=tokenizer(d['text'],add_special_tokens=False)['input_ids']
 for start in range(0,len(ids),1100):
  text=f'{row}:{d["id"]}:'+tokenizer.decode(ids[start:start+1200],skip_special_tokens=False);passages.append(text);source[text]=d
  if start+1200>=len(ids):break
(M/'linear_native_cache_location.json').write_text(json.dumps({'temporary_cache':str(O),'purpose':'GPU NER and embedding caches; final retrieval/protocol/validation persist in experiment directory'}))
inputs=[{'kind':'passage','hash_id':compute_mdhash_id(t,prefix='passage-'),'text':t} for t in passages]+[{'kind':'question','question_id':str(q['row_index']),'text':q['prompt']} for q in queries]
(O/'ner_inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inputs));save('protocol.json',{'method':'native LinearRAG','source_records':120238,'passages':len(passages),'chunk_tokens':1200,'overlap':100,'NER':'en_core_web_trf,8replicas/GPU','graph':'fresh native graph; no old graph or excluded-source NER reuse','top_k':5,'native_config':{'max_iterations':3,'top_k_sentence':1,'passage_ratio':.05,'passage_node_weight':.05,'damping':.5,'iteration_threshold':.4},'gold_opened':False})
(M/'linear_native_protocol.json').write_text((O/'protocol.json').read_text())
children=[]
for g in [0,1]:children.append(subprocess.Popen([sys.executable,str(M/'code/linear_ner_worker.py'),str(O),str(g)],env={**os.environ,'CUDA_VISIBLE_DEVICES':str(g),'HF_HUB_OFFLINE':'1'}))
codes=[c.wait() for c in children];assert codes==[0,0],codes
pe={};se={};qentities={}
for g in [0,1]:
 for r in rows(O/f'ner_gpu{g}.jsonl'):
  if r['kind']=='question':qentities[r['text']]=r['entities']
  else:
   pe[r['hash_id']]=r['passage_entities']
   for text,entities in r['sentence_entities'].items():se[text]=sorted(set(se.get(text,[]))|set(entities))
assert len(pe)==len(passages) and set(qentities)=={q['prompt'] for q in queries}
entities=sorted({e for values in pe.values() for e in values}|{e for values in qentities.values() for e in values});sentences=sorted(se)
namespaces=[('passage',passages),('entity',entities),('sentence',sentences),('question',[q['prompt'] for q in queries])];encoding=[];offsets={}
for kind,texts in namespaces:
 offsets[kind]=(len(encoding),len(texts));encoding.extend({'description':(cfg['query_prefix'] if kind=='question' else '')+text} for text in texts)
(O/'description_inputs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in encoding));(O/'embed.py').write_text((R/'analysis/near_duplicate_description_gpu_20261107/embed.py').read_text())
children=[subprocess.Popen(['__LOCAL_ASSET_ROOT__/miniconda3/envs/cyber-ft/bin/python',str(O/'embed.py')],env={**os.environ,'DESCRIPTION_GPU':str(g),'HF_HUB_OFFLINE':'1'}) for g in [0,1]];codes=[c.wait() for c in children];assert codes==[0,0],codes
vectors=np.lib.format.open_memmap(O/'all_vectors.npy',mode='w+',dtype='float32',shape=(len(encoding),2560))
for g in [0,1]:vectors[g::2]=np.load(O/f'vectors_gpu{g}.npy',mmap_mode='r')
I=O/'index';I.mkdir();schema=pa.schema([('hash_id',pa.large_string()),('text',pa.large_string()),('embedding',pa.list_(pa.float32()))])
lookup={}
for kind,texts in namespaces:
 start,count=offsets[kind]
 if kind=='question':
  lookup.update({text:np.array(vectors[start+i]) for i,text in enumerate(texts)});continue
 with pq.ParquetWriter(I/f'{kind}_embedding.parquet',schema) as writer:
  for b in range(0,count,256):
   part=texts[b:b+256];writer.write_table(pa.Table.from_pydict({'hash_id':[compute_mdhash_id(t,prefix=kind+'-') for t in part],'text':part,'embedding':np.asarray(vectors[start+b:start+b+len(part)]).tolist()},schema=schema))
 if kind=='entity':lookup.update({text:np.array(vectors[start+i]) for i,text in enumerate(texts)})
(I/'ner_results.json').write_text(json.dumps({'passage_hash_id_to_entities':pe,'sentence_to_entities':se}))
# All neural computation has finished on GPU. Native graph construction/traversal uses caches.
from src.config import LinearRAGConfig
import importlib
native=importlib.import_module("src.LinearRAG")
class Cache:
 embedding_dim=2560
 def encode(self,texts,**kwargs):
  if isinstance(texts,str):return lookup[texts]
  if not texts:return np.empty((0,2560),dtype=np.float32)
  assert all(t in lookup for t in texts),'Unexpected model inference while graph is running'
  return np.asarray([lookup[t] for t in texts])
class FrozenNER:
 def __init__(self,*args):pass
 def batch_ner(self,data,*args,**kwargs):assert not data;return {},{}
 def question_ner(self,text):return set(qentities[text])
native.SpacyNER=FrozenNER
config=LinearRAGConfig(dataset_name='index',embedding_model=Cache(),llm_model=None,working_dir=str(O),batch_size=8,max_workers=16,retrieval_top_k=5,max_iterations=3,top_k_sentence=1,passage_ratio=.05,passage_node_weight=.05,damping=.5,iteration_threshold=.4,use_vectorized_retrieval=False,enable_hybrid_attribute_fallback=False)
rag=native.LinearRAG(config);rag.index(passages);assert set(rag.passage_embedding_store.texts)==set(passages);results=rag.retrieve([{'question':q['prompt'],'answer':''} for q in queries]);assert len(results)==1000
sys.path.insert(0,'__LOCAL_ASSET_ROOT__/agentic-graphrag-benchmarks/src');from agentic_graphrag.benchmark.linearrag import LinearRAGSystem
out=[]
for q,result in zip(queries,results):
 selected=result['sorted_passage'];refs=[]
 for text in selected:
  d=source[text];cid=json.loads(d['raw_data'])['identifier'];refs.append({'cve_id':cid,'similarity':0,'text_override':text})
 record={'question_id':str(q['row_index']),'row_index':q['row_index'],'source_cve_id':q['cve_id'],'prompt':q['prompt'],'references':refs,'variable_references':True,'custom_messages':LinearRAGSystem.messages(q['prompt'],selected)}
 if not refs:record['pipeline_failure']='native graph returned no passages'
 out.append(record)
(M/'linearrag_prepared.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in out));save('graph_validation.json',{'fresh_graph':True,'vertices':rag.graph.vcount(),'edges':rag.graph.ecount(),'all_passages_from_filtered_records':True,'questions':1000,'gold_opened':False})

(M/'linear_native_validation.json').write_text((O/'graph_validation.json').read_text())
