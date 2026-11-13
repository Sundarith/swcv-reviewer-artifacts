"""Construct API clients from non-secret provider configuration."""
import os
from .api import ChatClient


def create_client(configuration, *, environ=None):
    """Select connection settings without changing prompts, retrieval, or scoring."""
    if configuration.get("backend", "openai_compatible") != "openai_compatible":
        raise ValueError("Unsupported API backend")
    if "api_key" in configuration:
        raise ValueError("Use api_key_env instead of storing a credential in configuration")
    env = os.environ if environ is None else environ
    return ChatClient(
        configuration["base_url"], configuration["model"],
        api_key=env.get(configuration.get("api_key_env", "SWCV_API_KEY")),
        timeout=configuration.get("timeout", 1800),
        token_parameter=configuration.get("token_parameter", "max_tokens"),
        omit_parameters=configuration.get("omit_parameters", ()),
        capabilities=configuration.get("capabilities", ()),
    )
