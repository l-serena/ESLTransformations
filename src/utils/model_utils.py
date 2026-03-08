from openai import OpenAI
import os


def return_model(model_config):
    """
    Use OpenAI API directly.
    Requires:
        export OPENAI_API_KEY=sk-...
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)
    return client