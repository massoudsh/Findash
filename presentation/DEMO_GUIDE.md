# Demo Guide for Stakeholders (Fundraising)

This guide gives you **three ways** to show the Octopus (Findash) platform to investors — from zero-install (best) to full local run.

---

## Option A: Live demo URL (recommended)

**Best for stakeholders: one link, no install.**

1. **Deploy the app** (you do this once):
   - **Frontend:** Deploy to [Vercel](https://vercel.com) (connect GitHub repo, deploy `frontend-nextjs` or root with correct build settings).
   - **Backend:** Deploy to [Render](https://render.com), [Railway](https://railway.app), or similar (use `Dockerfile.fastapi` or Python runtime; add PostgreSQL and Redis from their dashboards).
   - Set **environment variables** on both (see repo `env.example`). On the frontend, set `NEXT_PUBLIC_API_URL` to your backend URL (e.g. `https://your-api.onrender.com`).

2. **Share the frontend URL** with stakeholders, e.g.:
   - `https://findash-demo.vercel.app`  
   Or your custom domain.

3. **Stakeholders:** Open the link in a browser. No install, no dependencies.

**Suggested flow during the call:**  
Dashboard → Command Center (Options tab) → show decision tools (IV, Greeks, quick strategies) → Trade / Strategies → optionally Bots or Reports.

---

## Option B: One-command local run (Docker)

**For due diligence or when a live URL isn’t ready.** Stakeholders need Docker Desktop installed.

1. **Install:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac/Windows/Linux).

2. **Clone and run:**
   ```bash
   git clone https://github.com/massoudsh/Findash.git
   cd Findash
   docker compose -f docker-compose-core.yml up -d
   ```
   Wait 1–2 minutes for API, frontend, DB, and Redis to start.

3. **Open in browser:**
   - **App:** http://localhost:3000  
   - **API docs:** http://localhost:8011/docs  

4. **Stop when done:**
   ```bash
   docker compose -f docker-compose-core.yml down
   ```

**Note:** The frontend is built with `NEXT_PUBLIC_API_URL=http://localhost:8000` in the Dockerfile; the API is exposed on port **8011** on the host. If the app can’t reach the API, rebuild the frontend with:
`NEXT_PUBLIC_API_URL=http://localhost:8011` (see docker-compose `environment` for `frontend` and set it there, then rebuild).

---

## Option C: Local run without Docker

**If Docker isn’t an option** — requires Node.js, Python, and (optionally) PostgreSQL/Redis.

1. **Prerequisites:** Node.js 18+, Python 3.10+. Optional: PostgreSQL 14+, Redis.

2. **Backend:**
   ```bash
   cd Findash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   # Optional: set DATABASE_URL, REDIS_URL in .env (or leave unset for in-memory fallback)
   python3 start.py --reload
   ```
   API: http://localhost:8000

3. **Frontend (new terminal):**
   ```bash
   cd Findash/frontend-nextjs
   npm install
   echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
   npm run dev
   ```
   App: http://localhost:3000

4. **Stakeholder:** You share your screen, or they run the same steps and open http://localhost:3000.

---

## Option D: Google Drive / Colab (shareable package)

**To share a single folder with investors:**

1. **Create a package:**
   - Zip the repo (or a copy) **or** add a short `README_DEMO.txt` that links to:
     - This DEMO_GUIDE.md (raw GitHub or copy-paste)
     - Option A (live URL) if you have it
     - Option B (Docker one-command) and link to Docker Desktop

2. **Upload** the zip or the README + link to the repo to **Google Drive**, and share the folder link (view-only) with stakeholders.

3. **Google Colab:**  
   The main app is a **Next.js + FastAPI** stack, not a single notebook. For a “no-install” demo, **Option A (live URL)** is better. If you want a **Colab teaser**, you can add a separate notebook that, for example:
   - Runs a small Python script that calls your **deployed API** (e.g. `/health`, `/api/trading-bots/`) and prints results, or
   - Shows a minimal Streamlit/Gradio UI that talks to your live API.  
   We don’t include that notebook in this repo; you can add it under `presentation/` and link it from this guide.

---

## Quick reference

| Method              | Stakeholder install? | Best for              |
|---------------------|----------------------|------------------------|
| **A: Live URL**     | None                 | Pitch / fundraising   |
| **B: Docker**       | Docker Desktop       | Due diligence / local |
| **C: Local (no Docker)** | Node + Python  | Dev / custom setup    |
| **D: Drive + link** | None (if using A)    | Sharing materials     |

---

## What to show in the demo (suggested order)

1. **Dashboard** — Command center, Overview/Portfolio tabs, account cards, market status.
2. **Command Center** — Options tab first (decision tools: underlying, IV, Greeks, expiry, quick strategies).
3. **Trade** — Option terminal (or order flow); **Strategies** — strategy library.
4. **Market** tab — Real-time / market data.
5. **Trading Bots** — Create bot, start/pause, agent sources.
6. **Reports** (optional) — AI report / LLM status if enabled.

---

*Replace any placeholder URLs (e.g. `https://findash-demo.vercel.app`) with your actual live demo link before sharing.*
