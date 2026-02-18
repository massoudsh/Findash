# Fonts directory — Dana (and other local fonts)

**Put your Dana font files here** so the app can use them.

## Dana font (Persian / RTL)

Place your Dana font files in this folder, for example:

- `Dana.woff2` (preferred for modern browsers)
- `Dana.woff` (fallback)
- Optional weights: `Dana-Medium.woff2`, `Dana-Bold.woff2`, etc.

The app will load them via `@font-face` in `src/app/globals.css` and use **Dana** when the language is set to **Persian (فارسی)**.

## File names in code

In `globals.css` the paths used are:

- `/fonts/Dana.woff2`
- `/fonts/Dana.woff`

If your files have different names (e.g. `Dana-Regular.woff2`), either rename them to `Dana.woff2` / `Dana.woff` or update the `src: url(...)` paths in `globals.css` to match.

## Using Dana anywhere in the app

- **Automatic:** When the user selects Persian, the whole app uses the Dana font (via `[lang="fa"]` in CSS).
- **Manual:** Add the class `font-dana` to any element to use Dana regardless of language.
