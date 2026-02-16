"""
Report generation using open-source / free LLMs only (no paid API keys).

- Falcon (TGI): self-hosted via Docker, no key. FALCON_TGI_URL.
- FinGPT local: self-hosted via Docker, no key for public models. FINGPT_LOCAL_URL.
- FinGPT (HuggingFace): free HF token (https://huggingface.co/settings/tokens). HF_TOKEN, HF_API_URL.

Configure via env: FALCON_TGI_URL, FINGPT_LOCAL_URL, HF_API_URL + HF_TOKEN.
"""

import os
import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Env: Falcon via Hugging Face Text Generation Inference (Docker)
FALCON_TGI_URL = os.getenv("FALCON_TGI_URL", "").rstrip("/")  # e.g. http://localhost:8080

# Env: FinGPT via HuggingFace Inference API
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_FINGPT_MODEL = os.getenv("HF_FINGPT_MODEL", "FinGPT/fingpt-sentiment_llama2-13b_lora")
HF_API_URL = os.getenv("HF_API_URL", "https://api-inference.huggingface.co").rstrip("/")

# Env: FinGPT via local Docker inference server (docker-compose.llm.yml)
FINGPT_LOCAL_URL = os.getenv("FINGPT_LOCAL_URL", "").rstrip("/")  # e.g. http://localhost:8081


async def _generate_via_falcon_tgi(prompt: str, max_new_tokens: int = 1024) -> Optional[str]:
    """Generate completion via TII Falcon served with Hugging Face TGI (OpenAI-compatible)."""
    if not FALCON_TGI_URL:
        return None
    try:
        # TGI OpenAI-compatible endpoint
        url = f"{FALCON_TGI_URL}/v1/completions"
        payload = {
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": 0.3,
            "do_sample": True,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("text", "").strip()
            return None
    except Exception as e:
        logger.warning("Falcon TGI request failed: %s", e)
        return None


async def _generate_via_fingpt_local(prompt: str, max_new_tokens: int = 1024) -> Optional[str]:
    """Generate via local FinGPT inference container (Docker)."""
    if not FINGPT_LOCAL_URL:
        return None
    try:
        url = f"{FINGPT_LOCAL_URL}/generate"
        payload = {"inputs": prompt, "max_new_tokens": max_new_tokens, "temperature": 0.3}
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return (data.get("generated_text") or "").strip()
    except Exception as e:
        logger.warning("FinGPT local inference failed: %s", e)
        return None


async def _generate_via_fingpt_hf(prompt: str, max_new_tokens: int = 1024) -> Optional[str]:
    """Generate completion via FinGPT on HuggingFace Inference API."""
    if not HF_TOKEN:
        return None
    try:
        url = f"{HF_API_URL}/models/{HF_FINGPT_MODEL}"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.3,
                "return_full_text": False,
            },
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "").strip()
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
            return None
    except Exception as e:
        logger.warning("FinGPT HF inference failed: %s", e)
        return None


async def generate_report_text(prompt: str, max_new_tokens: int = 1024) -> str:
    """
    Generate report text using Falcon (TGI) or FinGPT (HF). Falls back to None
    so caller can use a mock/simulated response.
    """
    # Order: Falcon (TGI Docker) -> FinGPT local (Docker) -> FinGPT (HF API)
    out = await _generate_via_falcon_tgi(prompt, max_new_tokens)
    if out:
        return out
    out = await _generate_via_fingpt_local(prompt, max_new_tokens)
    if out:
        return out
    out = await _generate_via_fingpt_hf(prompt, max_new_tokens)
    if out:
        return out
    return ""
