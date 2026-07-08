'use client';

import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { CheckCircle2, ArrowLeft, Receipt } from 'lucide-react';
import { Suspense } from 'react';

function SuccessContent() {
  const params = useSearchParams();
  const orderId = params.get('id');
  const refId   = params.get('ref');

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-md text-center">
        {/* Icon */}
        <div className="flex justify-center mb-6">
          <div className="relative">
            <div className="absolute inset-0 rounded-full bg-green-500/20 animate-ping" />
            <div className="relative rounded-full bg-green-500/15 p-6 border border-green-500/30">
              <CheckCircle2 className="h-14 w-14 text-green-400" />
            </div>
          </div>
        </div>

        <h1 className="text-2xl font-bold text-foreground mb-2">پرداخت موفق!</h1>
        <p className="text-muted-foreground mb-8">تراکنش شما با موفقیت تأیید شد.</p>

        <div className="persian-card p-5 rounded-2xl mb-6 text-right space-y-3">
          {refId && (
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">کد رهگیری:</span>
              <span className="font-mono font-semibold text-green-400">{refId}</span>
            </div>
          )}
          {orderId && (
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">شماره سفارش:</span>
              <span className="font-mono text-foreground">{orderId}</span>
            </div>
          )}
          <div className="persian-accent-bar" />
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">وضعیت:</span>
            <span className="persian-badge">پرداخت‌شده</span>
          </div>
        </div>

        <div className="space-y-3">
          <Link
            href="/dashboard"
            className="btn-persian flex items-center justify-center gap-2 w-full rounded-2xl"
            style={{ height: '48px' }}
          >
            رفتن به داشبورد
            <ArrowLeft className="h-4 w-4" />
          </Link>
          {orderId && (
            <Link
              href={`/account?tab=payments`}
              className="flex items-center justify-center gap-2 w-full h-12 rounded-2xl border border-border text-muted-foreground hover:text-foreground hover:border-green-500/40 transition-colors text-sm"
            >
              <Receipt className="h-4 w-4" />
              مشاهده تاریخچه پرداخت
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}

export default function PaymentSuccessPage() {
  return (
    <Suspense fallback={<div className="min-h-screen flex items-center justify-center">در حال بارگذاری…</div>}>
      <SuccessContent />
    </Suspense>
  );
}
