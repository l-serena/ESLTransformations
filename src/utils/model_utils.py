import os

from openai import APIConnectionError, OpenAI


def check_local_server_reachable(client: OpenAI, model_name: str, port: int):
    """Verify the local vLLM/server is reachable; raise with a clear message if not."""
    try:
        client.models.list()
    except APIConnectionError as e:
        raise RuntimeError(
            f"Cannot reach the model server at {client.base_url}. "
            f"Ensure vLLM (or another OpenAI-compatible server) is running with --port {port}, "
            f"and that this script runs on the same machine (or set OPENAI_BASE_URL to the server address). "
            f"Original error: {e}"
        ) from e
    except Exception as e:
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            raise RuntimeError(
                f"Cannot connect to model server at {client.base_url}. "
                f"Start vLLM with --port {port} on this host, or set OPENAI_BASE_URL. Original: {e}"
            ) from e
        raise


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
    # If OPENAI_BASE_URL is set, use it (e.g. http://gpu-node:6001/v1 when vLLM runs on another host).
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    port = getattr(model_config, "port_num", 8000)
    if base_url:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        client = OpenAI(base_url=base_url, api_key="EMPTY")
    else:
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
    check_local_server_reachable(client, model_name, port)
    return client