'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { useSession } from 'next-auth/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface Plan {
  code: string;
  name_fa: string;
  price_toman: number;
  duration_days: number;
  features: string[] | null;
}

interface Status {
  active: boolean;
  plan_code: string | null;
  plan_name: string | null;
  end_at: string | null;
  auto_renew: boolean;
}

const toman = (value: number) => `${value.toLocaleString('fa-IR')} تومان`;

export default function SubscriptionPage() {
  const { status: sessionStatus } = useSession();
  const [plans, setPlans] = useState<Plan[]>([]);
  const [current, setCurrent] = useState<Status | null>(null);
  const [loading, setLoading] = useState(true);
  const [pendingCode, setPendingCode] = useState<string | null>(null);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const plansRes = await fetch('/api/subscriptions/plans', { cache: 'no-store' });
      if (!plansRes.ok) throw new Error('دریافت پلن‌ها ناموفق بود.');
      setPlans(await plansRes.json());
      const statusRes = await fetch('/api/subscriptions/me', { cache: 'no-store' });
      setCurrent(statusRes.ok ? await statusRes.json() : { active: false, plan_code: null, plan_name: null, end_at: null, auto_renew: false });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'ارتباط با سرور برقرار نشد.');
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => { void load(); }, [load]);

  async function subscribe(plan: Plan) {
    setPendingCode(plan.code);
    setError('');
    try {
      const response = await fetch('/api/subscriptions/subscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan_code: plan.code, callback_url: window.location.origin + '/payment/callback/zarinpal' }),
      });
      const result = await response.json();
      if (!response.ok || !result.redirect_url) throw new Error(typeof result.detail === 'string' ? result.detail : 'ایجاد درخواست پرداخت ناموفق بود.');
      window.location.href = result.redirect_url;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'ارتباط با سرور برقرار نشد.');
      setPendingCode(null);
    }
  }

  const signedOut = sessionStatus === 'unauthenticated';

  return (
    <section dir="rtl" className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">اشتراک من</h1>
        <p className="text-muted-foreground">وضعیت اشتراک و انتخاب پلن مناسب</p>
      </div>

      {signedOut && (
        <p role="alert" className="rounded-lg border p-4">
          برای مدیریت اشتراک ابتدا{' '}
          <Link className="underline" href="/auth/signin?callbackUrl=/account">وارد شوید</Link>.
        </p>
      )}
      {error && <p role="alert" className="text-destructive">{error}</p>}

      {loading ? <p role="status">در حال بارگذاری…</p> : <>
        <Card>
          <CardHeader><CardTitle>وضعیت فعلی</CardTitle></CardHeader>
          <CardContent>
            {current?.active ? (
              <div className="space-y-1">
                <p className="font-medium">پلن {current.plan_name ?? current.plan_code} فعال است</p>
                <p>پایان اشتراک: {current.end_at ? new Date(current.end_at).toLocaleDateString('fa-IR') : '—'}</p>
                <p className="text-sm text-muted-foreground">
                  تمدید خودکار: {current.auto_renew ? 'فعال' : 'غیرفعال'}
                </p>
              </div>
            ) : (
              <p>{signedOut ? 'برای مشاهده اشتراک وارد شوید.' : 'در حال حاضر اشتراک فعالی ندارید.'}</p>
            )}
          </CardContent>
        </Card>

        <div className="grid gap-4 md:grid-cols-3">
          {plans.map(plan => (
            <Card key={plan.code}>
              <CardHeader><CardTitle>{plan.name_fa}</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xl font-bold">{toman(plan.price_toman)}</p>
                <p className="text-sm text-muted-foreground">برای {plan.duration_days.toLocaleString('fa-IR')} روز</p>
                {plan.features && (
                  <ul className="list-disc space-y-1 pe-5 text-sm">
                    {plan.features.map(feature => <li key={feature}>{feature}</li>)}
                  </ul>
                )}
                <Button
                  className="w-full"
                  disabled={signedOut || pendingCode !== null || current?.plan_code === plan.code}
                  onClick={() => void subscribe(plan)}
                >
                  {pendingCode === plan.code ? 'در حال انتقال…' : current?.plan_code === plan.code ? 'پلن فعلی' : 'انتخاب و پرداخت'}
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </>}
    </section>
  );
}
