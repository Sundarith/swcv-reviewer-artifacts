"""CPU lexical retrieval for native IRCoT and Local CRAG, over the frozen CVE-only pool."""
import json,sys,re,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import bm25s
R=Path('__EXPERIMENT_WORKSPACE__');O=Path(sys.argv[1]);O.mkdir(parents=True,exist_ok=True);mask=json.loads((R/'analysis/ctibench1000_final_group_filters_20261107/source_supported_exclusions.json').read_text());eligible=set(mask['filtered_eligible_rows']);docs=[]
for i,l in enumerate((R/'logs/experiments/rcm335_nvd_first_fixed_pool_all_models_20261018_v1/shared/documents.jsonl').open()):
 if i in eligible:docs.append(json.loads(l))
assert len(docs)==120238
index=bm25s.BM25();index.index(bm25s.tokenize([d['text'] for d in docs],stopwords='en',show_progress=False));lock=threading.Lock()
class API(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):self.send_response(200);self.end_headers();self.wfile.write(b'{"ready":true}')
 def do_POST(self):
  try:
   request=json.loads(self.rfile.read(int(self.headers['Content-Length'])));k=int(request.get('k',20));assert 1<=k<=100
   with lock:ix,scores=index.retrieve(bm25s.tokenize([request['query']],stopwords='en',show_progress=False),k=k,show_progress=False)
   result=[{'document':docs[int(i)],'score':float(s)} for i,s in zip(ix[0],scores[0])]
   self.send_response(200);self.end_headers();self.wfile.write(json.dumps(result).encode())
  except Exception as e:self.send_response(500);self.end_headers();self.wfile.write(json.dumps({'error':str(e)}).encode())
(O/'sparse_service.json').write_text(json.dumps({'pid':__import__('os').getpid(),'port':8497,'documents':120238,'source':'rebuilt filtered pool only','stopwords':'en'}))
ThreadingHTTPServer(('127.0.0.1',8497),API).serve_forever()
