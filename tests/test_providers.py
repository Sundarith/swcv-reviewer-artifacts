import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from swcv.providers import create_client


@contextmanager
def endpoint(body):
    captured = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            captured.append({'path': self.path,
                             'request': json.loads(self.rfile.read(int(self.headers['Content-Length']))),
                             'authorization': self.headers.get('Authorization')})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class ProviderTests(unittest.TestCase):
    def test_provider_swap_preserves_messages(self):
        messages = [{'role': 'user', 'content': 'Original benchmark question?'}]
        answer = {'model': 'returned-version', 'choices': [{'message': {'content': 'CWE-79'}, 'finish_reason': 'stop'}]}
        with endpoint(answer) as (url, captured):
            for model, key_env in [('provider-one/model', 'FIRST_KEY'), ('provider-two/model', 'SECOND_KEY')]:
                client = create_client({'base_url': url, 'model': model, 'api_key_env': key_env}, environ={key_env: 'test-secret'})
                result = client.complete(messages)
                self.assertEqual(result['prediction'], 'CWE-79')
                self.assertEqual(result['response_model'], 'returned-version')
            self.assertEqual(captured[0]['request']['messages'], captured[1]['request']['messages'])
            self.assertEqual([x['request']['model'] for x in captured], ['provider-one/model', 'provider-two/model'])
            self.assertEqual(captured[0]['authorization'], 'Bearer test-secret')
            self.assertNotIn('test-secret', json.dumps(result))

    def test_explicit_token_mapping_and_parameter_omission(self):
        with endpoint({'choices': [{'message': {'content': 'answer'}}]}) as (url, captured):
            client = create_client({'base_url': url, 'model': 'model', 'token_parameter': 'max_completion_tokens', 'omit_parameters': ['temperature', 'n']}, environ={})
            client.complete([{'role': 'user', 'content': 'question'}], max_tokens=512)
            wire = captured[0]['request']
            self.assertEqual(wire['max_completion_tokens'], 512)
            self.assertNotIn('max_tokens', wire)
            self.assertNotIn('temperature', wire)
            self.assertNotIn('n', wire)

    def test_native_tool_and_probability_responses_preserved(self):
        calls = [{'id': 'call-1', 'type': 'function', 'function': {'name': 'search', 'arguments': '{"query":"test"}'}}]
        probabilities = {'content': [{'token': 'Yes', 'logprob': -0.3, 'top_logprobs': [{'token': 'No', 'logprob': -1.4}]}]}
        body = {'choices': [{'message': {'content': None, 'tool_calls': calls}, 'finish_reason': 'tool_calls', 'logprobs': probabilities}]}
        with endpoint(body) as (url, captured):
            client = create_client({'base_url': url, 'model': 'model', 'capabilities': ['tools', 'logprobs']}, environ={})
            tools = [{'type': 'function', 'function': {'name': 'search', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}}]
            result = client.complete([{'role': 'user', 'content': 'test'}], tools=tools, tool_choice='auto', logprobs=True, top_logprobs=20)
            self.assertEqual(result['tool_calls'], calls)
            self.assertEqual(result['logprobs'], probabilities)
            self.assertEqual(captured[0]['request']['tools'], tools)
            self.assertNotIn('response_format', captured[0]['request'])

    def test_unsupported_features_fail_before_network_request(self):
        client = create_client({'base_url': 'http://127.0.0.1:1/v1', 'model': 'model'}, environ={})
        with self.assertRaisesRegex(ValueError, 'capabilities'):
            client.complete([], tools=[])
        with self.assertRaisesRegex(ValueError, 'capabilities'):
            client.complete([], logprobs=True)
        with self.assertRaises(ValueError):
            client.complete([], top_logprobs=5)

    def test_invalid_config_does_not_construct_a_client(self):
        for extra in [{'api_key': 'secret'}, {'token_parameter': 'unknown'}, {'omit_parameters': ['max_tokens']}, {'backend': 'unknown'}]:
            with self.assertRaises(ValueError):
                create_client({'base_url': 'http://localhost/v1', 'model': 'model', **extra}, environ={})
