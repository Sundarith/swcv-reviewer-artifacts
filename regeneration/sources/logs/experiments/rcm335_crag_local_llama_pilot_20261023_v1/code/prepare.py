"""Both-GPU frozen evaluator, three-way local correction and strip refinement."""
import json,sys,subprocess,time,os
from pathlib import Path
from shared import P,S,rows,save,sha
from contracts import action,evidence_pool,strips,select_strips

def shard(gpu):
 import torch
 from transformers import AutoTokenizer,T5ForSequenceClassification
 torch.set_num_threads(4);assert torch.cuda.is_available();torch.cuda.set_device(gpu);torch.backends.cuda.matmul.allow_tf32=False
 device=f'cuda:{gpu}';tok=AutoTokenizer.from_pretrained(P/'evaluator',local_files_only=True)
 model,loading=T5ForSequenceClassification.from_pretrained(P/'evaluator',local_files_only=True,torch_dtype=torch.float32,output_loading_info=True)
 assert not loading.get('missing_keys') and not loading.get('unexpected_keys') and not loading.get('mismatched_keys'),loading
 assert model.config.num_labels==1;model=model.to(device).eval();model.config.use_cache=False
 docs=rows(S/'documents.jsonl');qs=rows(S/'queries.jsonl');dense=rows(S/'dense.jsonl');lexical=rows(S/'bm25.jsonl');byid={d['id']:i for i,d in enumerate(docs)}
 assert len(docs)==122745 and len(byid)==len(docs) and len(qs)==335
 def windows(q,text):
  qids=tok(q,add_special_tokens=False)['input_ids'];sep=tok(' [SEP] ',add_special_tokens=False)['input_ids']
  ts=tok(text,add_special_tokens=False)['input_ids'];result=[]
  # Explicit evaluator-only question windows cover every original token; reader receives full question.
  for qoffset in range(0,len(qids),192):
   pieceq=qids[qoffset:qoffset+256];prefix=pieceq+sep;budget=512-len(prefix)-1
   assert budget>=16
   for offset in range(0,len(ts),budget):
    piece=ts[offset:offset+budget];inp=prefix+piece+[tok.eos_token_id]
    assert len(inp)<=512 and inp[-1]==tok.eos_token_id
    result.append(dict(text=tok.decode(piece,skip_special_tokens=True),token_offset=offset,token_count=len(piece),query_token_offset=qoffset,query_token_count=len(pieceq),input_ids=inp))
   if qoffset+256>=len(qids):break
  return result
 def evaluate(items):
  result=[]
  for start in range(0,len(items),8):
   batch=items[start:start+8];encoded=tok.pad({'input_ids':[r['input_ids'] for r in batch]},padding=True,return_tensors='pt').to(device)
   with torch.inference_mode():scores=model(**encoded).logits.reshape(-1).float().cpu().tolist()
   assert all(__import__('math').isfinite(x) for x in scores)
   result.extend(scores)
  return result
 # Synthetic actual-GPU finite/deterministic scoring and exact no-truncation window gate.
 synthetic=windows('Which weakness affects memory after release?','A use after free memory access occurs. '+('Long evidence. '*900))
 assert sum(x['token_count'] for x in synthetic)==len(tok('A use after free memory access occurs. '+('Long evidence. '*900),add_special_tokens=False)['input_ids'])
 longq='Explain the underlying vulnerability mechanism. '*150
 longwindows=windows(longq,'A use after free access occurs. '*150)
 covered=set()
 for w in longwindows:covered.update(range(w['query_token_offset'],w['query_token_offset']+w['query_token_count']))
 assert covered==set(range(len(tok(longq,add_special_tokens=False)['input_ids'])))
 first=evaluate(synthetic[:2]);second=evaluate(synthetic[:2]);assert max(abs(a-b) for a,b in zip(first,second))<1e-5
 save(P/f'evaluator_gate_gpu{gpu}.json',dict(passed=True,gpu=gpu,device=device,full_FP32=True,strict_checkpoint_load=True,synthetic_scores=first,window_count=len(synthetic),no_CPU_fallback=True))
 out=S/f'correction_gpu{gpu}.jsonl';checkpoint=rows(out) if out.exists() else [];done={r['query_position'] for r in checkpoint}
 with out.open('a') as f:
  for i in range(gpu,335,2):
   if i in done:continue
   q=qs[i];assert q['row_index']==dense[i]['row_index']==lexical[i]['row_index'];assert q['prompt']==dense[i]['prompt']==lexical[i]['prompt']
   initial=[byid[x] for x in dense[i]['audit']['reference_ids']];alternate=[byid[x] for x in lexical[i]['audit']['reference_ids']];assert len(initial)==len(alternate)==5
   # Frozen source-excerpt policy: evaluate the same first256 words used by reader; record it.
   source={j:' '.join(docs[j]['text'].split()[:256]) for j in dict.fromkeys(initial+alternate)}
   doc_windows=[(j,w) for j in initial for w in windows(q['prompt'],source[j])];scores=evaluate([w for j,w in doc_windows]);doc_scores=[max(score for (j,w),score in zip(doc_windows,scores) if j==row) for row in initial]
   branch=action(doc_scores);pool=evidence_pool(branch,initial,alternate)
   units=[]
   for j in pool:
    for strip_index,strip in enumerate(strips(source[j])):
     for w in windows(q['prompt'],strip):units.append(dict(document_row=j,strip_index=strip_index,**w))
   values=evaluate(units)
   for u,v in zip(units,values):u['score']=v;u.pop('input_ids')
   dedup={}
   for u in units:
    key=(u['document_row'],u['strip_index'],u['text'])
    if key not in dedup or u['score']>dedup[key]['score']:dedup[key]=u
   selected=select_strips(list(dedup.values()))
   record=dict(query_position=i,row_index=q['row_index'],action=branch,initial_rows=initial,lexical_rows=alternate,initial_scores=doc_scores,selected=selected,all_strips=units,evaluator_source_words_max=256,gpu=gpu,at=time.time())
   f.write(json.dumps(record)+'\n');f.flush();done.add(i);save(P/f'preparation_gpu{gpu}.json',dict(done=len(done),total=len(range(gpu,335,2)),phase='corrective_retrieval',at=time.time()))

if __name__=='__main__':
 if len(sys.argv)>1:shard(int(sys.argv[1]));raise SystemExit
 children=[subprocess.Popen([sys.executable,'-u',__file__,str(g)],stdout=(P/f'prepare_gpu{g}.log').open('ab'),stderr=subprocess.STDOUT) for g in (0,1)]
 try:assert all(p.wait()==0 for p in children),'Evaluator preparation failed; inspect GPU logs'
 finally:
  for p in children:
   if p.poll() is None:p.terminate();p.wait()
 from shared import reader
 docs=rows(S/'documents.jsonl');qs=rows(S/'queries.jsonl');dense=rows(S/'dense.jsonl');byid={d['id']:d for d in docs};ranks=sorted(rows(S/'correction_gpu0.jsonl')+rows(S/'correction_gpu1.jsonl'),key=lambda x:x['query_position']);assert [x['query_position'] for x in ranks]==list(range(335))
 branches=__import__('collections').Counter()
 with (S/'corrected.jsonl').open('w') as out:
  for r,q,old in zip(ranks,qs,dense):
   check,_=reader(q,[byid[x] for x in old['audit']['reference_ids']],tag='dense');assert check==old['messages'],'Original reader parity failed'
   hits=[dict(docs[u['document_row']],text=u['text']) for u in r['selected']]
   messages,audit=reader(q,hits,tag='local_corpus_CRAG');assert messages[0]['content'].endswith(q['prompt']);branches[r['action']]+=1
   out.write(json.dumps(dict(q,messages=messages,audit=audit,action=r['action']))+'\n')
 save(P/'prepared_manifest.json',dict(passed=True,questions=335,original_reader_parity=335,branches=dict(branches),sha256={str(S/n):sha(S/n) for n in ['corrected.jsonl','correction_gpu0.jsonl','correction_gpu1.jsonl']},GPU_gates=[json.loads((P/f'evaluator_gate_gpu{g}.json').read_text()) for g in (0,1)]))
