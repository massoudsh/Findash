'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { useSession } from 'next-auth/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';

interface Policy {
  max_daily_drawdown_pct: number;
  max_position_concentration_pct: number;
  action_on_breach: string;
  enabled: boolean;
  updated_at: string | null;
}

interface Breach {
  id: number;
  rule: string;
  value: number;
  threshold: number;
  action_taken: string;
  created_at: string | null;
}

const ruleLabels: Record<string, string> = {
  max_daily_drawdown: 'افت روزانه',
  max_position_concentration: 'تمرکز روی یک دارایی',
};
const actionLabels: Record<string, string> = { alert: 'فقط هشدار', stop_bots: 'توقف ربات‌ها' };
const fa = (value: number) => value.toLocaleString('fa-IR', { maximumFractionDigits: 2 });

export default function RiskPolicyPage() {
  const { status } = useSession();
  const [policy, setPolicy] = useState<Policy | null>(null);
  const [breaches, setBreaches] = useState<Breach[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [policyRes, breachRes] = await Promise.all([
        fetch('/api/risk-policy/me', { cache: 'no-store' }),
        fetch('/api/risk-policy/breaches', { cache: 'no-store' }),
      ]);
      if (!policyRes.ok) throw new Error((await policyRes.json().catch(() => ({}))).detail ?? 'دریافت سیاست ریسک ناموفق بود.');
      setPolicy(await policyRes.json());
      setBreaches(breachRes.ok ? await breachRes.json() : []);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'ارتباط با سرور برقرار نشد.');
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => { void load(); }, [load]);

  async function save(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setSaving(true);
    setError('');
    setNotice('');
    try {
      const response = await fetch('/api/risk-policy/me', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          max_daily_drawdown_pct: Number(form.get('max_daily_drawdown_pct')),
          max_position_concentration_pct: Number(form.get('max_position_concentration_pct')),
          action_on_breach: form.get('action_on_breach'),
          enabled: form.get('enabled') === 'on',
        }),
      });
      if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail ?? 'ذخیره سیاست ناموفق بود.');
      setPolicy(await response.json());
      setNotice('سیاست ریسک ذخیره شد.');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'ارتباط با سرور برقرار نشد.');
    } finally {
      setSaving(false);
    }
  }

  const signedOut = status === 'unauthenticated';

  return (
    <main dir="rtl" className="mx-auto max-w-3xl space-y-6 p-6">
      <div>
        <h1 className="text-3xl font-bold">سیاست ریسک</h1>
        <p className="text-muted-foreground">حدهای ریسک خود را تعیین کنید تا در صورت عبور از آن‌ها هشدار بگیرید و ربات‌های معاملاتی متوقف شوند.</p>
      </div>

      {signedOut && (
        <p role="alert" className="rounded-lg border p-4">
          این تنظیمات مخصوص حساب شماست. برای مشاهده و تغییر، ابتدا{' '}
          <Link className="underline" href="/auth/signin?callbackUrl=/risk/policy">وارد شوید</Link>.
        </p>
      )}
      {error && <p role="alert" className="text-destructive">{error}</p>}
      {notice && <p role="status" className="text-green-600">{notice}</p>}

      {loading ? <p role="status">در حال بارگذاری…</p> : !signedOut && !error && policy && (
        <Card>
          <CardHeader><CardTitle>قوانین فعال</CardTitle></CardHeader>
          <CardContent>
            <form onSubmit={save} className="space-y-4">
              <label className="block">
                حداکثر افت روزانه (درصد)
                <Input name="max_daily_drawdown_pct" type="number" step="0.1" min="0.1" max="100" required defaultValue={policy.max_daily_drawdown_pct} className="mt-1" />
              </label>
              <label className="block">
                حداکثر تمرکز روی یک دارایی (درصد)
                <Input name="max_position_concentration_pct" type="number" step="0.1" min="0.1" max="100" required defaultValue={policy.max_position_concentration_pct} className="mt-1" />
              </label>
              <label className="block">
                اقدام هنگام نقض
                <select name="action_on_breach" defaultValue={policy.action_on_breach} className="mt-1 block w-full rounded border bg-background p-2">
                  {Object.entries(actionLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
                </select>
              </label>
              <label className="flex items-center gap-2">
                <input name="enabled" type="checkbox" defaultChecked={policy.enabled} />
                فعال بودن پایش خودکار
              </label>
              <Button type="submit" disabled={saving}>{saving ? 'در حال ذخیره…' : 'ذخیره'}</Button>
              <p className="text-sm text-muted-foreground">
                آخرین تغییر: {policy.updated_at ? new Date(policy.updated_at).toLocaleString('fa-IR') : 'ثبت نشده'}
              </p>
            </form>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader><CardTitle>تاریخچه نقض</CardTitle></CardHeader>
        <CardContent className="space-y-3">
          {breaches.length === 0 ? <p>تا این لحظه نقضی ثبت نشده است.</p> : breaches.map(breach => (
            <div key={breach.id} className="rounded-lg border p-3">
              <p className="font-medium">{ruleLabels[breach.rule] ?? breach.rule}</p>
              <p>مقدار {fa(breach.value)}٪ در برابر آستانه {fa(breach.threshold)}٪</p>
              <p className="text-sm text-muted-foreground">
                {actionLabels[breach.action_taken] ?? breach.action_taken} ·{' '}
                {breach.created_at ? new Date(breach.created_at).toLocaleString('fa-IR') : '—'}
              </p>
            </div>
          ))}
        </CardContent>
      </Card>
    </main>
  );
}
