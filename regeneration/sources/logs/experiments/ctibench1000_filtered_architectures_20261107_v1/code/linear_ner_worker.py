import os,sys,json,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
O=Path(sys.argv[1]);gpu=int(sys.argv[2])
def initialize():
 global ner
 import torch,spacy
 torch.set_num_threads(1);spacy.require_gpu(0);sys.path.insert(0,'__LOCAL_ASSET_ROOT__/agentic-graphrag-benchmarks/baselines/LinearRAG')
 from src.ner import SpacyNER
 ner=SpacyNER('en_core_web_trf')
def work(batch):
 result=[]
 for r,doc in zip(batch,ner.spacy_model.pipe([r['text'] for r in batch],batch_size=1)):
  if r['kind']=='question':result.append({**r,'entities':sorted({ent.text.lower() for ent in doc.ents if ent.label_ not in ['CARDINAL','ORDINAL']})})
  else:
   pe,se=ner.extract_entities_sentences(doc,r['hash_id']);result.append({**r,'passage_entities':pe[r['hash_id']],'sentence_entities':se})
 return result
if __name__=='__main__':
 rows=[json.loads(l) for l in (O/'ner_inputs.jsonl').open()][gpu::2]
 batches=[rows[i:i+8] for i in range(0,len(rows),8)]
 with (O/f'ner_gpu{gpu}.jsonl').open('x') as out,ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
  done=0
  for batch in pool.map(work,batches):
   for r in batch:out.write(json.dumps(r)+'\n')
   done+=len(batch);out.flush();(O/f'ner_progress_gpu{gpu}.json').write_text(json.dumps({'done':done,'total':len(rows),'workers':8}))
