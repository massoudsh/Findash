# Report Generation: Open-Source LLMs (Llama, Falcon & FinGPT)

The platform can generate reports using **only open-source or free-tier LLMs**. **No paid API keys are required** (no OpenAI, Anthropic, or other commercial APIs).

## LLM options: all free / open source

| Option | Model | API key / cost | Env |
|--------|--------|----------------|-----|
| **1. Llama (Ollama)** | Llama 3.2, Mistral, etc. | **None** – local via [Ollama](https://ollama.com) | `LLAMA_URL=http://localhost:11434`, `LLAMA_MODEL=llama3.2` |
| **2. Falcon (Docker)** | TII Falcon 7B Instruct | **None** – self-hosted, open source | `FALCON_TGI_URL=http://localhost:8080` |
| **3. FinGPT local (Docker)** | distilgpt2 (default) or FinGPT | **None** for public models; optional free **HF_TOKEN** for gated models | `FINGPT_LOCAL_URL=http://localhost:8081` |
| **4. FinGPT (HuggingFace Inference API)** | FinGPT or other HF models | **Free** – use a free [HuggingFace token](https://huggingface.co/settings/tokens); free tier has rate limits | `HF_TOKEN=hf_xxx` |

If none are set, the API falls back to a **simulated** response so the app still works.

## Options (pick one or combine)

| Option | Model | How | Env |
|--------|--------|-----|-----|
| **1. Llama (Ollama)** | Llama 3.2, Mistral, etc. | Local Ollama server | `LLAMA_URL`, `LLAMA_MODEL` |
| **2. Falcon (Docker)** | TII Falcon 7B Instruct | Text Generation Inference (TGI) | `FALCON_TGI_URL=http://localhost:8080` |
| **3. FinGPT local (Docker)** | FinGPT or small HF model | Optional inference container | `FINGPT_LOCAL_URL=http://localhost:8081` |
| **4. FinGPT (HuggingFace API)** | FinGPT on HF Inference | No Docker; free HF token only | `HF_TOKEN=hf_xxx` |

---

## 1. Llama via Ollama (recommended for report generation)

**No Docker required.** Install [Ollama](https://ollama.com), pull a model, then point the backend at it.

1. Install Ollama from [ollama.com](https://ollama.com) and start the server (e.g. `ollama serve` or run the app).
2. Pull a model, e.g. `ollama pull llama3.2` or `ollama pull mistral`.
3. Set in your API environment (e.g. `.env` or backend env):

```env
LLAMA_URL=http://localhost:11434
LLAMA_MODEL=llama3.2
```

- Report generation will call `POST {LLAMA_URL}/api/generate` with the chosen model.
- If the API runs in Docker and Ollama runs on the host, use the host URL (e.g. `http://host.docker.internal:11434` on Docker Desktop).

---

## 2. Falcon via Docker (TGI)

**GPU recommended** (e.g. 1x NVIDIA with ~16GB VRAM for 7B).

```bash
# From Findash repo root
docker compose -f docker-compose.llm.yml up -d tgi-falcon
```

Set in your API environment (e.g. `.env` or backend env):

```env
FALCON_TGI_URL=http://localhost:8080
```

- Completions: `POST http://localhost:8080/v1/completions` (OpenAI-compatible).
- To use from another host (e.g. API in Docker): use host DNS (e.g. `http://tgi-falcon:80` if API is on same compose network).

---

## 3. FinGPT-style local server (Docker)

Optional container that runs a small HuggingFace text-generation model (default `distilgpt2` for CPU; override for FinGPT).

```bash
docker compose -f docker-compose.llm.yml up -d fingpt-inference
```

Set:

```env
FINGPT_LOCAL_URL=http://localhost:8081
# Optional: use a FinGPT model (GPU recommended)
# HF_FINGPT_MODEL=FinGPT/fingpt-sentiment_llama2-13b_lora
# HF_TOKEN=hf_xxx
```

---

## 4. FinGPT via HuggingFace Inference API (free tier)

No Docker for the model; only a **free** HuggingFace token is required. Use a public model (e.g. `HF_FINGPT_MODEL=distilgpt2`) to avoid gated model terms.

1. Create a token at [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens).
2. Accept the model’s terms on the [FinGPT model page](https://huggingface.co/FinGPT/fingpt-sentiment_llama2-13b_lora) if gated.
3. Set:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Optional
HF_FINGPT_MODEL=FinGPT/fingpt-sentiment_llama2-13b_lora
HF_API_URL=https://api-inference.huggingface.co
```

---

## Backend behavior

- **Report endpoints** (e.g. `/llm/reports/generate-insights`, `/llm/reports/generate-comprehensive`) build a financial prompt and call the report LLM client.
- **Client order**: try **Llama (Ollama)** → **Falcon TGI** → **FinGPT local** → **FinGPT HF API**. First successful response is used.
- **Fallback**: if no endpoint is configured or all fail, the backend returns a simulated report so the UI still works.

---

## Env summary

| Variable | Meaning |
|----------|--------|
| `LLAMA_URL` | Ollama server URL (e.g. `http://localhost:11434`) |
| `LLAMA_MODEL` | Ollama model name (e.g. `llama3.2`, `mistral`); default `llama3.2` |
| `FALCON_TGI_URL` | Base URL of TGI (e.g. `http://localhost:8080`) |
| `FINGPT_LOCAL_URL` | Base URL of local FinGPT inference container (e.g. `http://localhost:8081`) |
| `HF_TOKEN` | HuggingFace API token for FinGPT (and optional local FinGPT container) |
| `HF_FINGPT_MODEL` | HF model id (e.g. `FinGPT/fingpt-sentiment_llama2-13b_lora`) |
| `HF_API_URL` | HF Inference API base (default `https://api-inference.huggingface.co`) |

See **config/env.example** in the repo for a full list of LLM-related variables (commented with no real keys).
