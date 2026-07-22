'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import { TrendingUp, Shield, CreditCard, AlertCircle } from 'lucide-react';

const PLANS = [
  { id: 'basic', label: 'پایه', price: 99000, features: ['تحلیل تکنیکال', 'اخبار بازار', 'هشدار قیمت'] },
  { id: 'pro',   label: 'حرفه‌ای', price: 249000, features: ['همه امکانات پایه', 'هوش مصنوعی معاملاتی', 'تحلیل آپشن', 'اسکن ریل‌تایم'] },
  { id: 'elite', label: 'الیت', price: 499000, features: ['همه امکانات حرفه‌ای', 'API اختصاصی', 'گزارش هفتگی AI', 'پشتیبانی اولویت‌دار'] },
];

export default function CheckoutPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [selected, setSelected] = useState('pro');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const plan = PLANS.find((p) => p.id === selected)!;

  async function handlePay() {
    if (!session) {
      router.push('/auth/signin?callbackUrl=/payment/checkout');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/payment/zarinpal/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          amount_toman: plan.price,
          description: `اشتراک فین دَش — پلن ${plan.label}`,
        }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || 'خطا در ایجاد تراکنش');
      }
      const data = await res.json();
      window.location.href = data.redirect_url;
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'خطای ناشناخته');
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-2xl">

        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-sm font-medium mb-4">
            <Shield className="h-3.5 w-3.5" />
            پرداخت امن از طریق زرین‌پال
          </div>
          <h1 className="text-3xl font-bold text-foreground mb-2">انتخاب اشتراک</h1>
          <p className="text-muted-foreground">دسترسی کامل به پلتفرم هوشمند معاملاتی</p>
        </div>

        {/* Plans */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
          {PLANS.map((p) => (
            <button
              key={p.id}
              onClick={() => setSelected(p.id)}
              className={[
                'text-right p-4 rounded-2xl border-2 transition-all duration-200',
                selected === p.id
                  ? 'border-green-500 bg-green-500/10 shadow-[0_0_16px_rgba(34,197,94,0.2)]'
                  : 'border-border bg-card hover:border-green-500/40',
              ].join(' ')}
            >
              <div className="flex items-center justify-between mb-3">
                <span className={['text-xs font-bold px-2 py-0.5 rounded-full', selected === p.id ? 'bg-green-500/20 text-green-400' : 'bg-muted text-muted-foreground'].join(' ')}>
                  {p.label}
                </span>
                <TrendingUp className={['h-4 w-4', selected === p.id ? 'text-green-400' : 'text-muted-foreground'].join(' ')} />
              </div>
              <div className="text-xl font-bold text-foreground mb-1">
                {p.price.toLocaleString('fa-IR')} تومان
              </div>
              <div className="text-xs text-muted-foreground mb-3">ماهانه</div>
              <ul className="space-y-1">
                {p.features.map((f) => (
                  <li key={f} className="text-xs text-muted-foreground flex items-center gap-1.5">
                    <span className={selected === p.id ? 'text-green-400' : 'text-muted-foreground'}>✓</span>
                    {f}
                  </li>
                ))}
              </ul>
            </button>
          ))}
        </div>

        {/* Summary */}
        <div className="persian-card p-6 rounded-2xl mb-6">
          <div className="flex justify-between items-center mb-4">
            <span className="text-muted-foreground">پلن انتخاب‌شده:</span>
            <span className="font-semibold text-foreground">{plan.label}</span>
          </div>
          <div className="persian-accent-bar mb-4" />
          <div className="flex justify-between items-center">
            <span className="text-muted-foreground">مبلغ قابل پرداخت:</span>
            <span className="text-2xl font-bold text-green-400">
              {plan.price.toLocaleString('fa-IR')} <span className="text-sm">تومان</span>
            </span>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3 mb-4 text-sm">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </div>
        )}

        {/* Auth notice */}
        {status === 'unauthenticated' && (
          <div className="flex items-center gap-2 text-yellow-400 bg-yellow-500/10 border border-yellow-500/20 rounded-xl px-4 py-3 mb-4 text-sm">
            <AlertCircle className="h-4 w-4 shrink-0" />
            برای پرداخت باید وارد حساب خود شوید.
          </div>
        )}

        {/* CTA */}
        <button
          onClick={handlePay}
          disabled={loading}
          className="w-full btn-persian flex items-center justify-center gap-2 h-13 text-base rounded-2xl disabled:opacity-60 disabled:cursor-not-allowed"
          style={{ height: '52px' }}
        >
          <CreditCard className="h-5 w-5" />
          {loading ? 'در حال اتصال به درگاه...' : 'پرداخت از طریق زرین‌پال'}
        </button>

        <p className="text-center text-xs text-muted-foreground mt-4">
          با پرداخت، <span className="text-green-400">قوانین استفاده</span> را می‌پذیرید.
          اطلاعات کارت شما توسط زرین‌پال محافظت می‌شود.
        </p>
      </div>
    </div>
  );
}
