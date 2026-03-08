from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os

from openai import OpenAI


def _is_local_openai_server(model_config) -> bool:
    """
    Heuristic: if you pass a port_num, assume you want the local OpenAI-compatible server.
    Otherwise use HuggingFace.
    """
    return getattr(model_config, "port_num", None) is not None


@dataclass
class HFGenerationClient:
    """
    Minimal OpenAI-like wrapper around a HuggingFace causal LM.

    Supports:
      - client.completions.create(model=..., prompt=[...], **sampling_params)
      - client.chat.completions.create(model=..., messages=[...], **sampling_params)

    Notes:
      - `model` arg is ignored (kept for API compatibility).
      - Uses tokenizer.apply_chat_template if available; otherwise a simple fallback.
    """
    model: Any
    tokenizer: Any
    device: Optional[str] = None

    class _ChoicesObj:
        def __init__(self, text: str):
            self.text = text
            self.message = type("Msg", (), {"content": text})

    class _RespObj:
        def __init__(self, texts: List[str]):
            self.choices = [HFGenerationClient._ChoicesObj(t) for t in texts]

    def _generate_one(self, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Do sampling only if temperature > 0
        do_sample = temperature is not None and float(temperature) > 0.0

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=float(temperature) if temperature is not None else 1.0,
                top_p=float(top_p) if top_p is not None else 1.0,
                max_new_tokens=int(max_tokens) if max_tokens is not None else 256,
            )

        # Decode only the newly generated tokens
        gen_ids = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def completions_create(self, model: str, prompt: List[str], **sampling_params):
        temperature = sampling_params.get("temperature", 0.2)
        top_p = sampling_params.get("top_p", 1.0)
        max_tokens = sampling_params.get("max_tokens", 256)

        texts = [self._generate_one(p, temperature, top_p, max_tokens) for p in prompt]
        return HFGenerationClient._RespObj(texts)

    def chat_completions_create(self, model: str, messages: List[Dict[str, str]], **sampling_params):
        # Convert messages to a single prompt
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: naive formatting
            prompt_lines = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                prompt_lines.append(f"{role.upper()}: {content}")
            prompt_lines.append("ASSISTANT:")
            prompt = "\n".join(prompt_lines)

        temperature = sampling_params.get("temperature", 0.2)
        top_p = sampling_params.get("top_p", 1.0)
        max_tokens = sampling_params.get("max_tokens", 256)

        text = self._generate_one(prompt, temperature, top_p, max_tokens)
        return HFGenerationClient._RespObj([text])

    @property
    def completions(self):
        return type("Completions", (), {"create": self.completions_create})

    @property
    def chat(self):
        return type("Chat", (), {"completions": type("ChatCompletions", (), {"create": self.chat_completions_create})})


def return_model(model_config, tokenizer=None):
    """
    If model_config.port_num is set -> connect to local OpenAI-compatible server (current behavior).
    Else -> load HuggingFace model directly and return an OpenAI-like wrapper.

    Requirements for HF mode:
      pip install transformers torch accelerate
      (optional) pip install bitsandbytes  # for 4-bit
      HF_TOKEN env var set if model is gated (Meta Llama)
    """
    if _is_local_openai_server(model_config):
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{model_config.port_num}/v1",
        )
        return client

    # HuggingFace direct mode
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_name = model_config.model_name

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ.get("HF_TOKEN", None),
            cache_dir=os.environ.get("MODEL_DIR", None),
        )

    # Optional: enable 4-bit if model_config has load_in_4bit=True
    load_in_4bit = bool(getattr(model_config, "load_in_4bit", False))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ.get("HF_TOKEN", None),
        cache_dir=os.environ.get("MODEL_DIR", None),
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=load_in_4bit,
    )
    model.eval()

    return HFGenerationClient(model=model, tokenizer=tokenizer)
