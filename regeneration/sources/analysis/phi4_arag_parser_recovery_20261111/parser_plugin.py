"""Phi JSON tool protocol with complete nested-array decoding for nonstreaming calls."""
import json

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import ExtractedToolCallInformation, FunctionCall, ToolCall
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.phi4mini_tool_parser import Phi4MiniJsonToolParser


@ToolParserManager.register_module('phi4_nested_json')
class Phi4NestedJsonToolParser(Phi4MiniJsonToolParser):
    def extract_tool_calls(self, model_output, request):
        marker = model_output.find('functools')
        if marker < 0:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        try:
            calls, _ = json.JSONDecoder().raw_decode(model_output[marker + len('functools'):].lstrip())
            if not isinstance(calls, list) or not calls:
                raise ValueError('Expected a nonempty call array')
            parsed = []
            for call in calls:
                arguments = call.get('arguments', call.get('parameters'))
                if not isinstance(call.get('name'), str) or not isinstance(arguments, dict):
                    raise ValueError('Expected a function name and object arguments')
                parsed.append(ToolCall(
                    id=make_tool_call_id(), type='function',
                    function=FunctionCall(name=call['name'], arguments=json.dumps(arguments, ensure_ascii=False)),
                ))
        except (ValueError, TypeError, AttributeError):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        return ExtractedToolCallInformation(tools_called=True, tool_calls=parsed, content=None)
