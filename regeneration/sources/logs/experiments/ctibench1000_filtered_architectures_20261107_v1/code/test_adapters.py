import ast,json,sys,tempfile,threading,unittest,gzip
from pathlib import Path
from types import SimpleNamespace as NS
R=Path('__EXPERIMENT_WORKSPACE__');O=Path(__file__).resolve().parents[1];sys.path[:0]=['__LOCAL_ASSET_ROOT__/agentic-graphrag-benchmarks/src',str(O/'code')]
class AdapterTests(unittest.TestCase):
 def test_native_ircot_uses_retrieval_and_original_question(self):
  from agentic_graphrag.benchmark.ircot import IRCoTSystem,IRCoTHit
  from agentic_graphrag.benchmark.schema import BenchmarkExample
  queries=[];calls=[]
  class Search:
   def search(self,text,*,top_k):queries.append((text,top_k));return (IRCoTHit('retained-reference','fictional','Fictional bounds-check evidence',1.),)
  class Chat:
   def generate(self,messages,**kw):
    calls.append((messages,kw));return NS(content='The answer is CWE-79.',finish_reason='stop',usage=NS(prompt_tokens=10,completion_tokens=5),endpoint='synthetic',response_id='synthetic',latency_seconds=0)
  system=IRCoTSystem(Search(),demonstration_prompt='Fictional demonstration',prompt_fitter=lambda x:x,first_sentence=lambda x:x,reasoning_stop=('\n','Q:'))
  example=BenchmarkExample(example_id='test',dataset='synthetic',question='Original synthetic question?',answers=(),passages=(),answerable=False)
  result=system.run(example,Chat());self.assertTrue(queries);self.assertTrue(calls);self.assertIn('CWE-79',result.prediction);self.assertTrue(all('Original synthetic question?' in messages[0]['content'] for messages,_ in calls));self.assertTrue(all(kw['max_tokens']==300 for _,kw in calls))
 def test_ragenta_fusion_exclusion_contract(self):
  from agentic_graphrag.benchmark.ragenta_diagnostic import fuse_hybrid_results
  hits=fuse_hybrid_results([('excluded','unusable',100.),('retained','usable',1.)],[('excluded','unusable',100.),('second','also usable',1.)],alpha=.65,top_k=20,exclude_ids={'excluded'})
  self.assertNotIn('excluded',{h.chunk_id for h in hits})
 def test_deferred_agent_replay_does_not_repeat_api_calls(self):
  tree=ast.parse((O/'code/agent_run.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Chat')
  with tempfile.TemporaryDirectory() as td:
   out=Path(td);local=threading.local();local.qid='1';local.call_index=0;local.saved=[];calls=[]
   response={'choices':[{'message':{'content':'synthetic'}}]}
   class HTTP:
    RequestException=RuntimeError
    @staticmethod
    def post(*a,**kw):calls.append(kw['json']);return NS(raise_for_status=lambda:None,json=lambda:response)
   def append(path,r):
    with gzip.open(path,'at') as f:f.write(json.dumps(r)+'\n')
   env={'local':local,'O':out,'cfg':{'context':2048,'chat_template':str(out/'template'),'template':{}},'tokenizer':NS(apply_chat_template=lambda *a,**kw:[1]),'Path':Path,'requests':HTTP,'append':append,'json':json,'time':__import__('time'),'NS':NS}
   (out/'template').write_text('synthetic');exec(compile(ast.Module(body=[node],type_ignores=[]),'agent_chat','exec'),env);client=env['Chat']();payload={'messages':[{'role':'user','content':'synthetic'}],'max_tokens':5}
   self.assertEqual(client.request(payload),response);self.assertEqual(len(calls),1)
   local.call_index=0;local.saved=[json.loads(l) for l in gzip.open(out/'requests_1.jsonl.gz','rt')]
   self.assertEqual(client.request(payload),response);self.assertEqual(len(calls),1)
   local.call_index=0
   with self.assertRaises(AssertionError):client.request({**payload,'max_tokens':6})
 def test_reader_generation_for_refined_and_native_messages(self):
  from make_queue import build
  import shutil
  for method in ['local_crag','linearrag']:
   p=build(method+'/_adapter_test',O/'naive_rag20_prepared.jsonl')
   try:
    for f in [*p.glob('runs/*/dense/code/run.py'),p/'code/preflight.py']:compile(f.read_text(),str(f),'exec')
    source=(p/'runs/granite30_fp8/dense/code/run.py').read_text();self.assertIn('text_override',source);self.assertIn('variable_references',source);self.assertIn("q.get('custom_messages')",source)
    if method=='linearrag':self.assertIn('Native LinearRAG reader prompt',source)
   finally:shutil.rmtree(p)
if __name__=='__main__':unittest.main()
