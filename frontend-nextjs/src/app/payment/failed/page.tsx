'use client';

import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { XCircle, RefreshCw, ArrowLeft } from 'lucide-react';
import { Suspense } from 'react';

const REASON_MAP: Record<string, string> = {
  cancelled:    'پرداخت توسط شما لغو شد.',
  user_cancelled: 'پرداخت توسط شما لغو شد.',
  verify_failed: 'تأیید تراکنش ناموفق بود.',
  not_found:    'سفارش یافت نشد.',
  '-9':  'اطلاعات ورودی نادرست',
  '-11': 'درگاه پرداخت غیرفعال است',
  '-22': 'تراکنش ناموفق بود',
  '-33': 'مبلغ مغایرت دارد',
  '-54': 'تراکنش منقضی شده',
};

function FailedContent() {
  const params = useSearchParams();
  const orderId = params.get('id');
  const reason  = params.get('reason') || '';
  const msg = REASON_MAP[reason] || 'پرداخت ناموفق بود. لطفاً دوباره تلاش کنید.';

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-md text-center">
        {/* Icon */}
        <div className="flex justify-center mb-6">
          <div className="rounded-full bg-red-500/10 p-6 border border-red-500/20">
            <XCircle className="h-14 w-14 text-red-400" />
          </div>
        </div>

        <h1 className="text-2xl font-bold text-foreground mb-2">پرداخت ناموفق</h1>
        <p className="text-muted-foreground mb-8">{msg}</p>

        {orderId && (
          <div className="persian-card p-4 rounded-2xl mb-6 text-right">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">شماره سفارش:</span>
              <span className="font-mono text-foreground">{orderId}</span>
            </div>
          </div>
        )}

        <div className="space-y-3">
          <Link
            href="/payment/checkout"
            className="btn-persian flex items-center justify-center gap-2 w-full rounded-2xl"
            style={{ height: '48px' }}
          >
            <RefreshCw className="h-4 w-4" />
            تلاش مجدد
          </Link>
          <Link
            href="/dashboard"
            className="flex items-center justify-center gap-2 w-full h-12 rounded-2xl border border-border text-muted-foreground hover:text-foreground hover:border-green-500/40 transition-colors text-sm"
          >
            <ArrowLeft className="h-4 w-4" />
            بازگشت به داشبورد
          </Link>
        </div>

        <p className="text-xs text-muted-foreground mt-6">
          در صورت کسر وجه از حساب بدون تأیید پرداخت،{' '}
          <Link href="/help" className="text-green-400 hover:underline">با پشتیبانی تماس بگیرید</Link>.
        </p>
      </div>
    </div>
  );
}

export default function PaymentFailedPage() {
  return (
    <Suspense fallback={<div className="min-h-screen flex items-center justify-center">در حال بارگذاری…</div>}>
      <FailedContent />
    </Suspense>
  );
}
