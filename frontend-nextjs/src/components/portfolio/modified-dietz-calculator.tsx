'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { modifiedDietz } from '@/lib/modified-dietz';

export function ModifiedDietzCalculator() {
  const [beginning, setBeginning] = useState('');
  const [ending, setEnding] = useState('');
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [flows, setFlows] = useState<{ amount: string; date: string }[]>([]);
  const inputClass = 'w-full rounded-md border bg-background p-2 text-sm';
  let result: ReturnType<typeof modifiedDietz> | null = null;
  let error = '';
  if (beginning && ending && start && end) {
    try {
      if (flows.some((flow) => !flow.amount || !flow.date)) {
        throw new Error('مبلغ و تاریخ همهٔ جریان‌های نقدی را تکمیل کنید.');
      }
      result = modifiedDietz(Number(beginning), Number(ending), Date.parse(start), Date.parse(end),
        flows.map((flow) => ({ amount: Number(flow.amount), timestamp: Date.parse(flow.date) })));
    } catch (cause) {
      error = cause instanceof Error ? cause.message : 'ورودی‌ها معتبر نیستند.';
    }
  }
  const number = (value: number) => value.toLocaleString('fa-IR', { maximumFractionDigits: 2 });

  return (
    <Card dir="rtl">
      <CardHeader><CardTitle>بازده با روش Modified Dietz (دیتز تعدیل‌شده)</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">محاسبهٔ مستقل با داده‌های واردشدهٔ شما؛ متصل به کیف پول نیست. ارزش‌ها شامل وجه نقد و دارایی‌ها و همهٔ مبالغ باید با یک واحد پول باشند.</p>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1 text-sm">ارزش ابتدای دوره<input className={inputClass} type="number" min="0" step="any" value={beginning} onChange={(e) => setBeginning(e.target.value)} /></label>
          <label className="space-y-1 text-sm">ارزش انتهای دوره<input className={inputClass} type="number" min="0" step="any" value={ending} onChange={(e) => setEnding(e.target.value)} /></label>
          <label className="space-y-1 text-sm">شروع دوره (میلادی)<input className={inputClass} type="date" value={start} onChange={(e) => setStart(e.target.value)} /></label>
          <label className="space-y-1 text-sm">پایان دوره (میلادی)<input className={inputClass} type="date" value={end} onChange={(e) => setEnd(e.target.value)} /></label>
        </div>
        <p className="text-sm text-muted-foreground">فقط واریز خارجی با مبلغ مثبت و برداشت خارجی با مبلغ منفی؛ خرید و فروش داخل پرتفوی جریان خارجی نیست. تاریخ‌ها ابتدای روز محاسبه می‌شوند؛ وزن جریان در شروع ۱ و در پایان صفر است.</p>
        {flows.map((flow, index) => (
          <div className="flex flex-wrap items-end gap-2" key={index}>
            <label className="flex-1 text-sm">مبلغ جریان {index + 1}<input className={inputClass} type="number" step="any" value={flow.amount} onChange={(e) => setFlows(flows.map((item, i) => i === index ? { ...item, amount: e.target.value } : item))} /></label>
            <label className="flex-1 text-sm">تاریخ جریان {index + 1}<input className={inputClass} type="date" min={start} max={end} value={flow.date} onChange={(e) => setFlows(flows.map((item, i) => i === index ? { ...item, date: e.target.value } : item))} /></label>
            <Button variant="outline" aria-label={`حذف جریان ${index + 1}`} onClick={() => setFlows(flows.filter((_, i) => i !== index))}>حذف</Button>
          </div>
        ))}
        <Button variant="outline" onClick={() => setFlows([...flows, { amount: '', date: '' }])}>افزودن واریز / برداشت</Button>
        <p className="text-xs text-muted-foreground" dir="ltr">R = (V₁ − V₀ − ΣCF) / (V₀ + ΣwCF), w = (end − flow date) / (end − start)</p>
        <div aria-live="polite" className="text-sm">
          {error ? <p role="alert" className="text-red-500">{error}</p> : result ? (
            <div className="space-y-2">
              <p>سود / زیان خالص از جریان نقدی: {number(result.gain)}</p>
              <p>سرمایهٔ موزون: {number(result.weightedCapital)}</p>
              <p className="font-bold">بازده دوره: {result.returnPercent === null ? 'قابل محاسبه نیست؛ سرمایهٔ موزون باید مثبت باشد.' : `${number(result.returnPercent)}٪`}</p>
              <p className="text-muted-foreground">این بازده سالانه‌شده نیست. داده‌های این محاسبه‌گر با خروج از صفحه حفظ نمی‌شوند.</p>
            </div>
          ) : <p>ارزش ابتدا، انتها و تاریخ‌های دوره را وارد کنید.</p>}
        </div>
      </CardContent>
    </Card>
  );
}
