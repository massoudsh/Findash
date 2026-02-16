"""
Minimal FinGPT-style inference server for Docker.
Uses HuggingFace pipeline; set HF_FINGPT_MODEL to a small model for CPU (e.g. distilgpt2)
or FinGPT model with GPU. Exposes POST /generate for report_llm_client.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_FINGPT_MODEL = os.getenv("HF_FINGPT_MODEL", "distilgpt2")  # small default for CPU; override to FinGPT model

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        from transformers import pipeline as hf_pipeline
        pipeline = hf_pipeline(
            "text-generation",
            model=HF_FINGPT_MODEL,
            token=HF_TOKEN or None,
            device=-1,  # CPU; set 0 for GPU
        )
    except Exception as e:
        pipeline = None
        print(f"Pipeline load failed: {e}. Report generation will return fallback.")
    yield
    pipeline = None


app = FastAPI(title="FinGPT-style Inference", lifespan=lifespan)


class GenerateRequest(BaseModel):
    inputs: str
    max_new_tokens: int = 512
    temperature: float = 0.3


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/generate")
def generate(req: GenerateRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        out = pipeline(
            req.inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=True,
            pad_token_id=pipeline.tokenizer.eos_token_id,
        )
        text = out[0]["generated_text"] if out else ""
        if text.startswith(req.inputs):
            text = text[len(req.inputs):].strip()
        return {"generated_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
