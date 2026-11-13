"""Fetch only public author-linked safetensors/data; never execute repository code."""
import hashlib,json,re,time
from pathlib import Path
from html.parser import HTMLParser
import requests
P=Path(__file__).resolve().parent.parent
FILES={'added_tokens.json':'1fiYYBunQGs3rQ14y28mCg5Ap_KKBCCDy','config.json':'1J2DCva5L6Mmx0JhzVWYiZfLg7YhRoY84','model.safetensors':'19p2bAYL2QxvUyloKv6udF0R69u1yie-H','special_tokens_map.json':'13LAXfli1JFO-Nan918RwYZnOpEB3nr4E','spiece.model':'132NL1aJIjdIUCCab6uk3R2EZuayTyy8F','tokenizer_config.json':'11GRPEmkDQ4h9EKFK5kb8OW0CEhXByyEd'}
class Form(HTMLParser):
 def __init__(self):super().__init__();self.fields={}
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if tag=='input' and d.get('name'):self.fields[d['name']]=d.get('value','')
manifest={}
for name,fileid in FILES.items():
 dest=P/'evaluator'/name
 if not dest.exists():
  session=requests.Session();url='https://drive.usercontent.google.com/download';r=session.get(url,params={'id':fileid,'export':'download'},stream=True,timeout=120);r.raise_for_status()
  if 'text/html' in r.headers.get('content-type',''):
   parser=Form();parser.feed(r.text);r.close();assert parser.fields.get('id')==fileid,parser.fields
   r=session.get(url,params=parser.fields,stream=True,timeout=120);r.raise_for_status()
  assert 'text/html' not in r.headers.get('content-type',''), 'Expected evaluator data, received HTML'
  temp=dest.with_suffix(dest.suffix+'.part')
  with temp.open('wb') as f:
   for block in r.iter_content(8*1024**2):f.write(block)
  r.close();temp.replace(dest)
 h=hashlib.sha256()
 with dest.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 manifest[name]=dict(drive_id=fileid,bytes=dest.stat().st_size,sha256=h.hexdigest());print(name,manifest[name],flush=True)
assert json.loads((P/'evaluator/config.json').read_text())['architectures']==['T5ForSequenceClassification']
(P/'evaluator_manifest.json').write_text(json.dumps(dict(source='Author README-linked Drive folder1CRFGsyNguXJwKSvFvJm_82GOOlkWSkW7',at=time.time(),files=manifest),indent=2)+'\n')
