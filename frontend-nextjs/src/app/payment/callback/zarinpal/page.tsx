'use client';

import { useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Loader2 } from 'lucide-react';
import { getBackendUrl } from '@/lib/backend-url';

/**
 * این صفحه فقط یک bridge است.
 * زرین‌پال callback را به frontend می‌فرستد؛
 * ما کاربر را به backend verify endpoint ریدایرکت می‌کنیم.
 */
function CallbackContent() {
  const params = useSearchParams();
  const router = useRouter();

  useEffect(() => {
    const status    = params.get('Status')    || params.get('status')    || '';
    const authority = params.get('Authority') || params.get('authority') || '';

    if (!authority) {
      router.replace('/payment/failed?reason=not_found');
      return;
    }

    // ریدایرکت به backend که verify را انجام می‌دهد
    const backendUrl = getBackendUrl();
    const target = `${backendUrl}/api/payment/zarinpal/callback?Status=${encodeURIComponent(status)}&Authority=${encodeURIComponent(authority)}`;
    window.location.href = target;
  }, [params, router]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-4">
      <Loader2 className="h-10 w-10 text-green-400 animate-spin" />
      <p className="text-muted-foreground">در حال تأیید پرداخت...</p>
    </div>
  );
}

export default function ZarinPalCallbackPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-10 w-10 text-green-400 animate-spin" />
      </div>
    }>
      <CallbackContent />
    </Suspense>
  );
}
