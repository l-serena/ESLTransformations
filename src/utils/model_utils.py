from openai import OpenAI
import os


def return_model(model_config):
    """
    Return an OpenAI client for either:
    - OpenAI-hosted models (gpt-*, o1-*, o3-*) via the default OpenAI API (or OPENAI_BASE_URL), OR
    - a local OpenAI-compatible server (e.g. vLLM) at http://localhost:{port}/v1 for HF model ids.

    OpenAI-hosted requires:
        export OPENAI_API_KEY=sk-...
    """
    model_name = getattr(model_config, "model_name", None) or ""

    is_openai_hosted = model_name.startswith(("gpt-", "o1-", "o3-"))

    if is_openai_hosted:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)

    # Default: local OpenAI-compatible endpoint (vLLM etc.)
    port = getattr(model_config, "port_num", 8000)
    return OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")