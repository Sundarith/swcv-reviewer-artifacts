import json,hashlib
from pathlib import Path
from functools import lru_cache
P=Path(__file__).resolve().parent.parent
S=P/'shared'
ROOT=Path('__EXPERIMENT_WORKSPACE__')
OLD=ROOT/'logs/experiments/granite4_2_8b_ircot_linearrag_cna335_20261013_v1'
def rows(p):return [json.loads(x) for x in Path(p).open()]
def save(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(r,indent=2));tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
@lru_cache()
def tokenizers():
 from transformers import AutoTokenizer
 return {name:AutoTokenizer.from_pretrained(cfg['snapshot'],local_files_only=True) for name,cfg in json.loads((P/'reader_models.json').read_text()).items()}
def counts(messages):return {name:len(t.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True)) for name,t in tokenizers().items()}
def vote_text(v):return '\n'.join(f'{label}: {weight:.17g}' for label,weight in sorted(v['weights'].items(),key=lambda x:-x[1]))
def reader(q,hits=(),votes=None,tag=''):
 # All methods receive identical source snippets and questions across models.
 budget=256
 while True:
  sections=[]
  for i,r in enumerate(hits,1):
   meta=json.loads(r['raw_data']);header=f"[{i}] {meta['identifier']}: {meta.get('name','')}"
   if meta.get('cwe_ids'):header+='\nStructured source CWE mappings: '+', '.join(meta['cwe_ids'])
   text=r['text'];text=text.split('\n## ',1)[0] if meta['source']=='CWE' else text
   sections.append(header+'\n'+' '.join(text.split()[:budget]))
  text=''
  if sections:text+='Retrieved reference evidence:\n'+'\n\n'.join(sections)+'\n\n'
  if votes is not None:text+='CWE similarity-weighted vote totals from20 mapped CVEs (not probabilities):\n'+vote_text(votes)+'\n\n'
  text+='Use the available evidence and your knowledge to identify the CWE. End with only the CWE ID.\n\n'+q['prompt']
  messages=[{'role':'user','content':text}];n=counts(messages)
  if max(n.values())<=6000:break
  assert budget>0,'Question and evidence headers alone exceed common budget'
  budget=budget//2
 return messages,{'input_tokens_by_model':n,'words_per_reference_max':budget,'reference_ids':[r['id'] for r in hits],'excerpted':True,'tag':tag}
