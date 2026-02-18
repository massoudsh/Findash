# Stakeholder Presentation & Demo (Fundraising)

This folder contains everything you need to present the Octopus (Findash) platform to stakeholders and run a friendly demo.

---

## Contents

| File | Purpose |
|------|---------|
| **STAKEHOLDER_DECK.md** | Slide-by-slide content for your pitch. Copy into **Google Slides** or **PowerPoint**, then add your logos, amounts, and contact details. |
| **index.html** | Simple in-browser slide deck. Open in a browser or host on GitHub Pages / any static host. Use **arrow keys** or the buttons to navigate. |
| **DEMO_GUIDE.md** | How to give the **demo**: live URL (recommended), one-command Docker run, or local run. Share this with stakeholders who want to try the app themselves. |
| **README.md** | This file. |

---

## Quick links for stakeholders

1. **Presentation (slides)**  
   - Use **STAKEHOLDER_DECK.md** in Google Slides or PowerPoint, **or**  
   - Open **index.html** in a browser (or share a link if you host it).

2. **Demo (using the app)**  
   - **Best:** Share a **live URL** (e.g. frontend on Vercel + backend on Render). See **DEMO_GUIDE.md** → Option A.  
   - **Alternative:** Share **DEMO_GUIDE.md** and the repo link; they can run the one-command Docker demo (Option B) or clone and run locally (Option C).

3. **Google Drive / Colab**  
   - Upload this **presentation** folder (or the whole repo zip) to Google Drive and share the link.  
   - In the folder, include **DEMO_GUIDE.md** (or a short **README_DEMO.txt** with the live demo URL and link to DEMO_GUIDE).  
   - **Colab:** The main app is Next.js + FastAPI, so a “no-install” browser link (Option A) is better than Colab for the full app. You can add a separate Colab notebook for an analytics/API teaser if needed.

---

## One-command demo (you or stakeholders)

From the **repo root** (not inside `presentation/`):

```bash
./scripts/demo-stakeholders.sh
```

Then open **http://localhost:3000** in a browser.  
Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

---

## Before the meeting

- [ ] Replace placeholder **live demo URL** in STAKEHOLDER_DECK.md, index.html, and DEMO_GUIDE.md with your deployed link (if you use Option A).
- [ ] Add your **contact**, **raise amount**, and **use of funds** in the deck.
- [ ] Rehearse the demo flow: Dashboard → Command Center (Options) → decision tools → Trade/Strategies → Bots (optional).

---

*Repo: [github.com/massoudsh/Findash](https://github.com/massoudsh/Findash)*
