const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { createRequire } = require('node:module');
const ts = require('typescript');
const filename = `${__dirname}/utils.ts`;
const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS },
});
const exportsUnderTest = {};
new Function('exports', 'require', compiled.outputText)(exportsUnderTest, createRequire(filename));
test('landing page renders its feature directory without undefined references', () => {
  const filename = `${__dirname}/../app/page.tsx`;
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  });
  const page = {};
  new Function('exports', 'require', compiled.outputText)(page, createRequire(filename));
  const React = require('react');
  const { renderToStaticMarkup } = require('react-dom/server');
  const html = renderToStaticMarkup(React.createElement(page.default));
  assert.ok(html.includes('پایش بازار'));
  assert.ok(html.includes('مدیریت پرتفوی'));
  assert.ok(html.includes('href="/auth/signup"'));
});

const { formatTomanValue, safeCallbackPath } = exportsUnderTest;
const origin = 'https://app.example';

test('toman value stays unchanged in toman and converts tenfold in rial', () => {
  assert.equal(formatTomanValue(1234, 'IRT'), `${(1234).toLocaleString('fa-IR')} تومان`);
  assert.equal(formatTomanValue(1234, 'IRR'), `${(12340).toLocaleString('fa-IR')} ریال`);
});
test('negative profit converts without losing sign', () => {
  assert.equal(formatTomanValue(-50, 'IRR'), `${(-500).toLocaleString('fa-IR')} ریال`);
  assert.equal(formatTomanValue(0, 'IRT'), '۰ تومان');
});
test('login preserves same-origin checkout path and query', () => {
  assert.equal(safeCallbackPath('/payment/checkout?plan=pro', origin), '/payment/checkout?plan=pro');
  assert.equal(safeCallbackPath(`${origin}/dashboard?tab=portfolio`, origin), '/dashboard?tab=portfolio');
});
test('login defaults safely on missing, external or malformed redirect', () => {
  for (const value of [null, '', 'https://other.example', '//other.example', 'javascript:alert(1)', 'https://[', `${origin}//other.example`]) {
    assert.equal(safeCallbackPath(value, origin), '/dashboard');
  }
});
