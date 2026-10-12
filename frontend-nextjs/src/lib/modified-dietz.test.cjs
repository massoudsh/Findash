const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const ts = require('typescript');
const moduleUnderTest = { exports: {} };
const source = fs.readFileSync(`${__dirname}/modified-dietz.ts`, 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } });
new Function('exports', compiled.outputText)(moduleUnderTest.exports);
const { modifiedDietz } = moduleUnderTest.exports;

const calculate = (beginning, ending, flows = []) => modifiedDietz(beginning, ending, 0, 100, flows);

test('no flows reduces to simple period return', () => {
  assert.equal(calculate(1000, 1100).returnPercent, 10);
});
test('mid-period contribution is time weighted', () => {
  const result = calculate(1000, 1650, [{ amount: 500, timestamp: 50 }]);
  assert.equal(result.gain, 150);
  assert.equal(result.weightedCapital, 1250);
  assert.equal(result.returnPercent, 12);
});
test('withdrawals are negative external flows', () => {
  assert.equal(calculate(1000, 640, [{ amount: -400, timestamp: 50 }]).returnPercent, 5);
});
test('boundary flows receive weights one and zero', () => {
  const result = calculate(1000, 1700, [{ amount: 500, timestamp: 0 }, { amount: 200, timestamp: 100 }]);
  assert.equal(result.weightedCapital, 1500);
  assert.equal(result.returnPercent, 0);
});
test('multiple unsorted flows and negative performance', () => {
  const result = calculate(1000, 950, [{ amount: -100, timestamp: 75 }, { amount: 200, timestamp: 25 }]);
  assert.equal(result.gain, -150);
  assert.equal(result.weightedCapital, 1125);
  assert.ok(Math.abs(result.returnPercent + 13.333333333333334) < 1e-10);
});
test('nonpositive capital is unavailable, not zero return', () => {
  assert.equal(calculate(0, 0).returnPercent, null);
  assert.equal(calculate(100, 0, [{ amount: -200, timestamp: 0 }]).returnPercent, null);
});
test('initial funding can establish weighted capital', () => {
  assert.equal(calculate(0, 110, [{ amount: 100, timestamp: 0 }]).returnPercent, 10);
});
test('reject invalid dates, values, periods and out-of-period flows', () => {
  for (const value of [NaN, Infinity, -1]) assert.throws(() => calculate(value, 100));
  assert.throws(() => modifiedDietz(100, 100, 10, 10, []));
  assert.throws(() => modifiedDietz(100, 100, 20, 10, []));
  for (const timestamp of [-1, 101, NaN]) assert.throws(() => calculate(100, 100, [{ amount: 10, timestamp }]));
  assert.throws(() => calculate(100, 100, [{ amount: Infinity, timestamp: 50 }]));
});
