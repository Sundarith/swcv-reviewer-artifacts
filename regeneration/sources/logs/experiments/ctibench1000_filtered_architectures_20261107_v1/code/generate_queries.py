"""Model-dependent first stages; no gold, returned-answer retries, or answer repair."""
import json,sys,time,threading,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path('__EXPERIMENT_WORKSPACE__');sys.path.insert(0,'__LOCAL_ASSET_ROOT__/swcv-reproducibility/src')
from swcv.api import ChatClient
from transformers import AutoTokenizer,TokenizersBackend
profile_path=Path(sys.argv[1]);method=sys.argv[2];out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=False)
profile=json.loads(profile_path.read_text());loader=TokenizersBackend if profile.get('tokenizer_loader')=='TokenizersBackend' else AutoTokenizer;tokenizer=loader.from_pretrained(profile['tokenizer_path'],local_files_only=True)
Q=R/'logs/experiments/rcm1000_source_priority_all_models_20261017_v1/shared/queries.jsonl';queries=[json.loads(l) for l in Q.open()]
H=R/'logs/experiments/rcm335_hyde_fusion_nonreasoning_full335_20261024_v1/native'
if method=='hyde':template=(H/'hyde/hypothesis_prompt.txt').read_text()
elif method=='fusion':template=json.loads((H/'fusion/rewrite_prompt.json').read_text())
elif method=='etr':template=(R/'logs/experiments/rcm1000_source_priority_all_models_20261017_v1/extraction_prompt.txt').read_text()
else:raise ValueError(method)
client=ChatClient('http://127.0.0.1:8480/v1',profile['model_name']);lock=threading.Lock()
prepared=[]
for q in queries:
 messages=[{'role':m['role'],'content':m['content'].format(question=q['prompt'])} for m in template] if isinstance(template,list) else [{'role':'user','content':template.format(question=q['prompt'])}]
 ids=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,chat_template=Path(profile['chat_template']).read_text(),**profile['template'])
 assert len(ids)+512<=profile['context'],('Intermediate context limit',method,q['row_index'],len(ids))
 prepared.append((q,messages))
(out/'input_freeze.json').write_text(json.dumps({'n':1000,'method':method,'profile_sha256':hashlib.sha256(profile_path.read_bytes()).hexdigest(),'template':template,'temperature':0,'output_cap':512,'workers':16,'gold_opened':False},indent=2)+'\n')
def one(pair):
 q,messages=pair;record={'question_id':str(q['row_index']),'row_index':q['row_index'],**client.complete(messages,max_tokens=512,temperature=0)}
 with lock,(out/'outputs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 return record
with ThreadPoolExecutor(max_workers=16) as pool:
 for count,f in enumerate(as_completed([pool.submit(one,p) for p in prepared]),1):
  f.result();(out/'progress.json').write_text(json.dumps({'done':count,'total':1000}))
f=out/'outputs.jsonl';(out/'prediction_freeze.json').write_text(json.dumps({'rows':1000,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'gold_opened':False})+'\n')
