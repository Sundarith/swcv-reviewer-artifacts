import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from swcv.api import ChatClient
from swcv.inference import prepare

class APIContractTests(unittest.TestCase):
    def invoke(self, responses, extra_body=None):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                captured.append((self.path, self.rfile.read(int(self.headers['Content-Length'])), self.headers.get('Authorization')))
                status, data = responses[min(len(captured)-1, len(responses)-1)]
                self.send_response(status); self.end_headers(); self.wfile.write(data.encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        waits = []
        try:
            client = ChatClient(f'http://127.0.0.1:{server.server_port}/v1', 'test-model', 'synthetic-key', sleep=waits.append)
            result = client.complete([{'role':'user','content':'test'}], extra_body=extra_body)
        finally:
            server.shutdown(); server.server_close(); thread.join()
        return result, captured, waits

    def test_exception_retry_identical_and_terminal_truncation(self):
        answer = json.dumps({'model':'test-model-version','choices':[{'message':{'content':'CWE-79'},'finish_reason':'length'}],'usage':{'completion_tokens':512}})
        result, calls, waits = self.invoke([(503,'unavailable'), (200,answer)])
        self.assertEqual(result['prediction'], 'CWE-79'); self.assertEqual(result['finish_reason'], 'length')
        self.assertEqual(len(calls),2); self.assertEqual(calls[0][1],calls[1][1]); self.assertEqual(waits,[1])
        self.assertEqual(calls[0][0],'/v1/chat/completions'); self.assertEqual(calls[0][2],'Bearer synthetic-key')
        self.assertNotIn('synthetic-key',json.dumps(result))

    def test_malformed_success_not_retried(self):
        result, calls, waits = self.invoke([(200,'not JSON')])
        self.assertEqual(len(calls),1); self.assertEqual(waits,[]); self.assertIn('returned_response_error',result)

    def test_exhausted_requests_remain_empty(self):
        result, calls, waits = self.invoke([(500,'failed')])
        self.assertEqual(len(calls),4); self.assertEqual(waits,[1,2,4]); self.assertEqual(result['prediction'],'')
        self.assertIn('request_error',result)

    def test_empty_answer_not_retried(self):
        answer=json.dumps({'choices':[{'message':{'content':''},'finish_reason':'stop'}]})
        result,calls,_=self.invoke([(200,answer)])
        self.assertEqual(len(calls),1); self.assertEqual(result['prediction'],'')

    def test_prompt_preserves_question_and_rejects_missing_evidence(self):
        question='Original question?\nKeep spacing.'
        refs=[{'doc_id':str(i),'title':'CVE reference','text':'x'*700} for i in range(20)]
        prompt={'SYSTEM':'system','PROMPT':'{question}\n{candidates}'}
        output=prepare([{'question_id':'1','question':question,'references':refs,'gold':'do not transmit'}],prompt)
        self.assertTrue(output[0]['messages'][1]['content'].startswith(question))
        self.assertNotIn('do not transmit',json.dumps(output))
        self.assertEqual(output[0]['messages'][1]['content'].count('x'*600),20)
        with self.assertRaises(ValueError):prepare([{'question_id':'1','question':question,'references':refs[:19]}],prompt)

    def test_runner_freezes_complete_ledger_without_gold(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from swcv.inference import main
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if 'GOLD_SENTINEL' in json.dumps(body):
                    self.send_response(400);self.end_headers();return
                self.send_response(200);self.end_headers()
                self.wfile.write(json.dumps({'model':'mock-version','choices':[{'message':{'content':'CWE-79'},'finish_reason':'stop'}]}).encode())
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'inputs.jsonl';prompt=root/'prompt.json';out=root/'run';config=root/'config.json'
                source.write_text(json.dumps({'question_id':'1','question':'Synthetic question','gold':'GOLD_SENTINEL'})+'\n')
                prompt.write_text(json.dumps({'SYSTEM':'Synthetic test system'}))
                config.write_text(json.dumps({'backend':'openai_compatible','model':'configured-model','workers':3,'token_parameter':'max_completion_tokens'}))
                argv=['swcv-infer','--config',str(config),'--inputs',str(source),'--out',str(out),'--base-url',f'http://127.0.0.1:{server.server_port}/v1','--model','mock','--method','closed_book','--prompt',str(prompt),'--workers','2']
                with patch('sys.argv',argv),patch.dict('os.environ',{'SWCV_API_KEY':''}):main()
                self.assertEqual(json.loads((out/'manifest.json').read_text())['status'],'completed')
                self.assertEqual(json.loads((out/'manifest.json').read_text())['model_requested'],'mock')
                self.assertEqual(json.loads((out/'manifest.json').read_text())['workers'],2)
                self.assertEqual(json.loads((out/'prediction_freeze.json').read_text())['count'],1)
                self.assertEqual(json.loads((out/'predictions.jsonl').read_text())['prediction'],'CWE-79')
                transmitted = json.loads((out/'predictions.jsonl').read_text())['request']
                self.assertEqual(transmitted['max_completion_tokens'],512)
                self.assertNotIn('max_tokens',transmitted)
                self.assertNotIn('GOLD_SENTINEL',(out/'prepared.jsonl').read_text())
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_provider_options_forwarded_without_schema_constraints(self):
        answer=json.dumps({'id':'synthetic-generation','provider':'synthetic-provider','choices':[{'message':{'content':'CWE-79'},'finish_reason':'stop'}]})
        options={'provider':{'only':['synthetic-provider'],'allow_fallbacks':False,'require_parameters':True},'reasoning':{'enabled':False}}
        result,calls,_=self.invoke([(200,answer)],extra_body=options)
        payload=json.loads(calls[0][1])
        self.assertEqual(payload['provider'],options['provider'])
        self.assertEqual(payload['reasoning'],options['reasoning'])
        self.assertNotIn('response_format',payload)
        self.assertEqual(result['generation_id'],'synthetic-generation')
        self.assertEqual(result['provider'],'synthetic-provider')
