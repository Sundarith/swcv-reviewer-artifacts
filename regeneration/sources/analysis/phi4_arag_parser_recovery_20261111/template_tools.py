"""Prepare a local tokenizer view of OpenAI-style tool history."""
import json
from copy import deepcopy


def tokenizer_messages(messages):
    """Match vLLM's template input while preserving the original HTTP request."""
    rendered = deepcopy(messages)
    for message in rendered:
        if message['role'] != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            function = call['function']
            arguments = function.get('arguments')
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError('Tool arguments must decode to a JSON object')
            function['arguments'] = arguments
    return rendered
