"""Regression checks for nonstreaming JSON tool parsing."""
import json
from pathlib import Path
from parser_plugin import Phi4NestedJsonToolParser

parser = Phi4NestedJsonToolParser.__new__(Phi4NestedJsonToolParser)
cases = [
    [{'name': 'lookup', 'arguments': {'query': 'hello'}}],
    [{'name': 'chunk_read', 'arguments': {'chunk_ids': ['fixture-1', 'fixture-2']}}],
    [{'name': 'chunk_read', 'arguments': {'chunk_ids': ['literal]inside']}},
     {'name': 'lookup', 'arguments': {'query': '[nested] string'}}],
    [{'name': 'lookup', 'parameters': {'query': 'hello'}}],
]
for calls in cases:
    output = parser.extract_tool_calls('functools' + json.dumps(calls), None)
    assert output.tools_called and len(output.tool_calls) == len(calls)
    for expected, actual in zip(calls, output.tool_calls, strict=True):
        assert expected['name'] == actual.function.name
        assert expected.get('arguments', expected.get('parameters')) == json.loads(actual.function.arguments)
for text in ('ordinary answer', 'functools[{"broken":', 'functools[]', 'functools[2]'):
    output = parser.extract_tool_calls(text, None)
    assert not output.tools_called and output.content == text
Path(__file__).with_name('parser_tests.json').write_text(json.dumps({
    'passed': True, 'valid_cases': 4, 'invalid_or_ordinary_cases': 4,
    'protocol': 'Same functools JSON-array protocol; nonstreaming only',
}, indent=2) + '\n')
print('8 parser regression checks passed')
