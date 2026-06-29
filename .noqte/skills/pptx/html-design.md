# Premium decks & reports — design in HTML, render with `noqte-render`

**This is the PREFERRED way to make a visually polished presentation or report**
(especially Persian/RTL, or anything with charts). You design in HTML+CSS — where
RTL is native, colors are 100% yours, and charts are crisp inline SVG — then:

```bash
noqte-render deck.html /project/outputs/<name>.pdf      # PDF (best, default)
noqte-render deck.html /project/outputs/<name>.pptx     # PPTX — each slide a
                                                        # full-bleed image of the
                                                        # SAME design (premium,
                                                        # not editable text)
```

Design ONCE; the PDF and the PPTX look identical. Make the PPTX too **only if the
user wants a PowerPoint file** — otherwise PDF is the deliverable.

`OK <path>` = success. The in-app preview renders the PDF directly.

## Why HTML (not python-pptx / pptxgenjs)

- **RTL is correct for free** — `dir="rtl"` puts bullets/markers on the RIGHT, no
  bidi fights.
- **Colors are fully dynamic** — you choose a palette that fits THIS topic via CSS
  variables; nothing is hardcoded.
- **Charts never get cut** — inline `<svg>` scales perfectly; no cropped images.
- **Real design** — gradients, layered layout, big type, generous whitespace.

## Rules that make the render correct

1. One self-contained `.html` file. **Inline all CSS** (`<style>`). No external
   network (fonts/JS/images must be local or data-URIs).
2. Persian font is installed: `font-family: 'Vazirmatn', sans-serif;`.
3. **Always** set `print-color-adjust: exact` (below) or backgrounds print white.
4. Deck = 1280×720 pages; report = A4. One `.slide` / page per printed page with
   `page-break-after: always`.
5. Embed a project image with a relative/absolute path as `<img src="...">` and
   `object-fit: cover|contain` (contain for charts/diagrams so nothing is cut).
6. **Pick a palette for the topic** (finance→deep green/navy, energy→teal/amber,
   food→warm clay/crimson, tech→indigo/blue…). Set it once in `:root`.

## Premium RTL deck boilerplate (adapt freely)

```html
<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><style>
  /* ——— pick colors for THE TOPIC; this example = teal/amber energy theme ——— */
  :root{
    --bg:#0E1B1E; --panel:#13262A; --ink:#F4F8F7; --muted:#9DB2B0;
    --accent:#19A7A0; --accent2:#E0A340; --line:rgba(255,255,255,.12);
  }
  @page{ size:1280px 720px; margin:0; }
  *{ box-sizing:border-box; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
  html,body{ margin:0; padding:0; font-family:'Vazirmatn',sans-serif; color:var(--ink); }
  .slide{ width:1280px; height:720px; position:relative; overflow:hidden;
          background:var(--bg); page-break-after:always; padding:84px 96px; }
  .kicker{ color:var(--accent2); font-size:20px; font-weight:700; letter-spacing:.02em; }
  h1{ font-size:80px; line-height:1.05; margin:.15em 0; font-weight:800; }
  h2{ font-size:46px; line-height:1.1; margin:0 0 28px; font-weight:800; }
  .sub{ color:var(--muted); font-size:26px; }
  .rule{ width:96px; height:7px; background:var(--accent); border-radius:6px; margin:20px 0; }
  ul{ list-style:none; margin:0; padding:0; }
  li{ font-size:27px; line-height:1.55; margin:18px 0; padding-right:40px; position:relative; color:#E7EEED; }
  li::before{ content:""; position:absolute; right:0; top:.62em; width:13px; height:13px;
              background:var(--accent); border-radius:3px; transform:rotate(45deg); } /* marker on the RIGHT */
  .grid{ display:flex; gap:32px; align-items:center; height:100%; }
  .col{ flex:1; } .col.media{ flex:1.05; }
  .card{ background:var(--panel); border:1px solid var(--line); border-radius:20px; padding:28px 32px; }
  .stat{ font-size:64px; font-weight:800; color:var(--accent); }
  .footer{ position:absolute; bottom:34px; left:96px; right:96px; display:flex;
           justify-content:space-between; color:var(--muted); font-size:16px; }
  .dot{ width:11px; height:11px; border-radius:50%; background:var(--accent); display:inline-block; }
  img.fill{ width:100%; height:100%; object-fit:cover; border-radius:18px; }
</style></head><body>

  <!-- COVER -->
  <section class="slide" style="display:flex;flex-direction:column;justify-content:center;
       background:radial-gradient(120% 120% at 100% 0%, #16323A 0%, var(--bg) 60%);">
    <div class="kicker">گزارشِ راهبردی</div>
    <div class="rule"></div>
    <h1>عنوانِ بزرگِ ارائه</h1>
    <div class="sub">زیرعنوانِ توضیحی در یک خط</div>
    <div class="footer"><span>تهیه‌شده توسط دستیار نقطه · ۱۴۰۵</span><span><span class="dot"></span></span></div>
  </section>

  <!-- CONTENT + BULLETS (markers auto on the right via dir=rtl) -->
  <section class="slide">
    <h2>یافته‌های کلیدی</h2><div class="rule"></div>
    <ul>
      <li>نکتهٔ اول با توضیحِ کوتاه.</li>
      <li>نکتهٔ دوم که کمی بلندتر است و در صورت لزوم به خط بعد می‌رود.</li>
      <li>نکتهٔ سوم.</li>
    </ul>
    <div class="footer"><span>عنوانِ ارائه · ۲</span><span><span class="dot"></span></span></div>
  </section>

  <!-- CONTENT + INLINE SVG CHART (never cut, scales perfectly) -->
  <section class="slide">
    <h2>روندِ فروش</h2><div class="rule"></div>
    <div class="grid">
      <div class="col">
        <ul><li>رشدِ پیوسته در چهار فصل.</li><li>اوجِ فروش در فصلِ سوم.</li></ul>
      </div>
      <div class="col media">
        <svg viewBox="0 0 520 320" width="100%" style="direction:ltr">
          <g font-family="Vazirmatn" font-size="16" fill="var(--muted)">
            <line x1="60" y1="20" x2="60" y2="270" stroke="var(--line)"/>
            <line x1="60" y1="270" x2="500" y2="270" stroke="var(--line)"/>
            <!-- bars: height ∝ value -->
            <rect x="100" y="150" width="64" height="120" rx="6" fill="var(--accent)"/>
            <rect x="200" y="100" width="64" height="170" rx="6" fill="var(--accent)"/>
            <rect x="300" y="60"  width="64" height="210" rx="6" fill="var(--accent2)"/>
            <rect x="400" y="120" width="64" height="150" rx="6" fill="var(--accent)"/>
            <text x="132" y="292" text-anchor="middle">بهار</text>
            <text x="232" y="292" text-anchor="middle">تابستان</text>
            <text x="332" y="292" text-anchor="middle">پاییز</text>
            <text x="432" y="292" text-anchor="middle">زمستان</text>
          </g>
        </svg>
      </div>
    </div>
    <div class="footer"><span>عنوانِ ارائه · ۳</span><span><span class="dot"></span></span></div>
  </section>

</body></html>
```

Notes:
- For a **light** theme just flip the vars (`--bg:#FAF9F5; --ink:#1A1915; --muted:#6E675C`).
  Keep ONE accent dominant; add a second only as a highlight.
- For **stat slides** use big `.stat` numbers in cards. Vary layouts across slides.
- Charts: simple bar/line/pie are easy as inline SVG. Compute bar heights/pie
  arcs from the data. This guarantees they match the palette and never crop.

## Premium RTL report (A4) boilerplate — for PDF reports

For a **report/PDF** (not slides), use A4 normal flow. Render with
`noqte-render report.html /project/outputs/<name>.pdf`. For an editable **Word**
file from the SAME html: `noqte-render report.html /project/outputs/<name>.docx`
(LibreOffice keeps the RTL text + structure; styling is simpler than the PDF).
Or use `noqte-make` for a native RTL .docx with right-side bullets.

```html
<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><style>
  :root{ --ink:#1A1915; --body:#3B3630; --muted:#7C746A; --accent:#C15F3C; --line:#E6E1D6; --soft:#FAF7F3; }
  @page{ size:A4; margin:20mm 18mm; }
  *{ box-sizing:border-box; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
  body{ margin:0; font-family:'Vazirmatn',sans-serif; color:var(--body); font-size:11.5pt; line-height:1.9; }
  .title{ font-size:30pt; font-weight:800; color:var(--ink); margin:0; line-height:1.2; }
  .subtitle{ color:var(--muted); font-size:13pt; margin:.3em 0 0; }
  .rule{ height:4px; width:90px; background:var(--accent); border-radius:4px; margin:14px 0 26px; }
  h2{ font-size:16pt; font-weight:800; color:var(--ink); margin:26px 0 8px;
      padding-right:12px; border-right:4px solid var(--accent); }
  h3{ font-size:12.5pt; font-weight:700; color:var(--ink); margin:16px 0 6px; }
  p{ margin:0 0 10px; text-align:justify; }
  ul{ list-style:none; margin:0 0 12px; padding:0; }
  li{ position:relative; padding-right:22px; margin:7px 0; }
  li::before{ content:""; position:absolute; right:0; top:.7em; width:8px; height:8px;
              background:var(--accent); border-radius:2px; transform:rotate(45deg); } /* marker on the RIGHT */
  .callout{ background:var(--soft); border:1px solid var(--line); border-radius:12px; padding:14px 18px; margin:14px 0; }
  table{ width:100%; border-collapse:collapse; margin:14px 0; font-size:10.5pt; }
  th,td{ border:1px solid var(--line); padding:8px 10px; text-align:right; }
  th{ background:var(--soft); color:var(--ink); font-weight:700; }
  .src{ color:var(--muted); font-size:9.5pt; }
</style></head><body>
  <div class="title">عنوانِ گزارش</div>
  <div class="subtitle">زیرعنوانِ توضیحی · ۱۴۰۵</div>
  <div class="rule"></div>

  <h2>خلاصهٔ مدیریتی</h2>
  <p>پاراگرافِ مقدماتیِ راست‌چین و موجّه…</p>
  <div class="callout">نکتهٔ کلیدی در یک کادرِ برجسته.</div>

  <h2>یافته‌ها</h2>
  <ul><li>مورد اول.</li><li>مورد دوم.</li></ul>
  <table>
    <tr><th>شاخص</th><th>مقدار</th></tr>
    <tr><td>نرخِ رشد</td><td>۱۲٪</td></tr>
  </table>

  <!-- inline SVG charts exactly like the deck section -->
</body></html>
```

## QA (required)

Render, convert to images, inspect with fresh eyes (a subagent):

```bash
noqte-render deck.html /tmp/qa.pdf
pdftoppm -jpeg -r 130 /tmp/qa.pdf /tmp/qa
```

Check: text not overflowing the 720px height, markers on the RIGHT, colors
printed (not white), charts uncut, consistent margins. Fix and re-render.
