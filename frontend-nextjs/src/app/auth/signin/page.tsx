"use client";

import { safeCallbackPath } from '@/lib/utils';
import Link from "next/link";
import { Suspense, useState, type FormEvent } from "react";
import { signIn } from 'next-auth/react';
import { useRouter, useSearchParams } from 'next/navigation';
import { ArrowLeft, ShieldCheck, Sparkles, TrendingUp } from "lucide-react";

function SignInForm() {
  const router = useRouter();
  const params = useSearchParams();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError('');
    const form = new FormData(event.currentTarget);
    const callbackUrl = safeCallbackPath(params.get('callbackUrl'), window.location.origin);
    try {
      const result = await signIn('credentials', {
        email: String(form.get('email')),
        password: String(form.get('password')),
        redirect: false,
        callbackUrl,
      });
      if (!result?.ok || result.error) {
        setError('ورود ناموفق بود؛ اطلاعات حساب و اتصال به سرور را بررسی کنید.');
        return;
      }
      router.push(callbackUrl);
      router.refresh();
    } catch {
      setError('اتصال برقرار نشد. دوباره تلاش کنید.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen grid lg:grid-cols-2 persian-pattern-bg">
      {/* Visual side */}
      <section className="hidden lg:flex items-center justify-center p-10 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-green-500/20 via-transparent to-transparent" />
        <div className="relative max-w-md w-full persian-card persian-border p-8">
          <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-green-500/10 border border-green-500/20 mb-6">
            <TrendingUp className="h-6 w-6 text-green-400" />
          </div>
          <h2 className="text-3xl font-black mb-3">به اختاپوس خوش آمدید</h2>
          <p className="text-muted-foreground leading-7 mb-6">داشبورد فارسی، سریع و امن برای پایش بازار و مدیریت ریسک.</p>
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-2xl bg-green-500/10 border border-green-500/20 p-4">
              <div className="text-2xl font-black text-green-400">۷۱۲</div>
              <div className="text-xs text-muted-foreground">امتیاز اعتباری</div>
            </div>
            <div className="rounded-2xl bg-card/70 border border-border p-4">
              <div className="text-2xl font-black text-foreground">۳۴</div>
              <div className="text-xs text-muted-foreground">ریسک زنده</div>
            </div>
          </div>
        </div>
      </section>

      {/* Form side */}
      <section className="flex items-center justify-center px-4 py-10">
        <div className="w-full max-w-md persian-card p-6 sm:p-8 rounded-3xl">
          <div className="text-center mb-8">
            <div className="inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-green-500/10 border border-green-500/20 mb-4">
              <ShieldCheck className="h-7 w-7 text-green-400" />
            </div>
            <h1 className="text-2xl font-black mb-2">ورود به حساب</h1>
            <p className="text-sm text-muted-foreground leading-7">برای پرداخت و امکانات حساب وارد شوید؛ داشبورد نمونه همچنان عمومی است.</p>
          </div>
          {params.get('signup') === 'success' && <p className="mb-4 text-green-500">ثبت‌نام انجام شد؛ وارد حساب شوید.</p>}
          <form onSubmit={handleSubmit} className="space-y-4">
            <label className="block">ایمیل<input name="email" type="email" autoComplete="email" required dir="ltr" className="mt-1 w-full rounded-xl border bg-background p-3" /></label>
            <label className="block">رمز عبور<input name="password" type="password" autoComplete="current-password" required dir="ltr" className="mt-1 w-full rounded-xl border bg-background p-3" /></label>
            {error && <p role="alert" className="text-sm text-red-500">{error}</p>}
            <button disabled={loading} className="btn-persian w-full rounded-xl p-3 disabled:opacity-50">{loading ? 'در حال ورود…' : 'ورود'}</button>
          </form>
          <Link href={`/auth/signup?${new URLSearchParams({ callbackUrl: params.get('callbackUrl') || '/dashboard' })}`} className="mt-4 block text-center text-sm">ساخت حساب جدید</Link>

          <Link href="/dashboard" className="btn-persian w-full h-12 rounded-2xl mt-6 flex items-center justify-center gap-2">
            مشاهده داشبورد نمونه
            <ArrowLeft className="h-4 w-4" />
          </Link>

          <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-xs text-muted-foreground space-y-2">
            <p className="font-semibold text-foreground/80 flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-green-400" />
              داده‌ها نمایشی هستند
            </p>
            <p className="leading-6">برای دیدن نمای کلی، پرتفولیو، بازار، معاملات و تحلیل نیازی به حساب کاربری نیست.</p>
          </div>
        </div>
      </section>
    </main>
  );
}

export default function SignInPage() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center">در حال بارگذاری…</div>}>
      <SignInForm />
    </Suspense>
  );
}
