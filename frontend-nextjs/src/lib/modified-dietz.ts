export interface CashFlow {
  amount: number;
  timestamp: number;
}

export function modifiedDietz(
  beginningValue: number,
  endingValue: number,
  start: number,
  end: number,
  cashFlows: CashFlow[],
): { gain: number; weightedCapital: number; returnPercent: number | null } {
  if (![beginningValue, endingValue, start, end].every(Number.isFinite) || end <= start) {
    throw new Error('ارزش‌ها و تاریخ‌ها باید معتبر و پایان دوره پس از شروع باشد.');
  }
  if (beginningValue < 0 || endingValue < 0) {
    throw new Error('ارزش ابتدا و انتهای پرتفوی نمی‌تواند منفی باشد.');
  }
  let netFlow = 0;
  let weightedCapital = beginningValue;
  for (const flow of cashFlows) {
    if (!Number.isFinite(flow.amount) || !Number.isFinite(flow.timestamp) || flow.timestamp < start || flow.timestamp > end) {
      throw new Error('مبلغ جریان نقدی باید معتبر و تاریخ آن در محدودهٔ دوره باشد.');
    }
    netFlow += flow.amount;
    weightedCapital += flow.amount * ((end - flow.timestamp) / (end - start));
  }
  const gain = endingValue - beginningValue - netFlow;
  const returnPercent = weightedCapital > 0 ? (gain / weightedCapital) * 100 : null;
  if (![gain, weightedCapital].every(Number.isFinite) || (returnPercent !== null && !Number.isFinite(returnPercent))) {
    throw new Error('مقادیر واردشده خارج از محدودهٔ محاسبه هستند.');
  }
  return { gain, weightedCapital, returnPercent };
}
