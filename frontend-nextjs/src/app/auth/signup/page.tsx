"use client";

import Link from "next/link";
import { useState } from "react";
import { ArrowLeft, LockKeyhole, Mail, ShieldCheck, UserPlus } from "lucide-react";

export default function SignUpPage() {
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError("");
    setLoading(true);
    const form = e.currentTarget;
    const email = (form.elements.namedItem("email") as HTMLInputElement).value;
    const password = (form.elements.namedItem("password") as HTMLInputElement).value;
    const confirm = (form.elements.namedItem("confirm") as HTMLInputElement).value;
    const firstName = (form.elements.namedItem("firstName") as HTMLInputElement).value || "کاربر";
    const lastName = (form.elements.namedItem("lastName") as HTMLInputElement).value || "جدید";

    if (password !== confirm) {
      setError("رمز عبور و تأیید آن یکسان نیستند");
      setLoading(false);
      return;
    }

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8011";
      const res = await fetch(`${apiUrl}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email,
          password,
          confirm_password: confirm,
          first_name: firstName,
          last_name: lastName,
        }),
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        setError(data.message || data.detail || "خطا در ثبت‌نام. لطفاً دوباره تلاش کنید.");
        setLoading(false);
        return;
      }
      window.location.href = "/auth/signin?signup=success";
    } catch {
      setError("اتصال به سرور برقرار نشد. لطفاً اینترنت خود را بررسی کنید.");
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen grid lg:grid-cols-2 persian-pattern-bg">
      <section className="hidden lg:flex items-center justify-center p-10 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-green-500/20 via-transparent to-transparent" />
        <div className="relative max-w-md w-full persian-card persian-border p-8">
          <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-green-500/10 border border-green-500/20 mb-6">
            <UserPlus className="h-6 w-6 text-green-400" />
          </div>
          <h2 className="text-3xl font-black mb-3">حساب معاملاتی خود را بسازید</h2>
          <p className="text-muted-foreground leading-7 mb-6">شروع سریع با داشبورد فارسی، امتیاز اعتباری و مدیریت ریسک زنده.</p>
          <div className="space-y-3">
            {['داشبورد موبایل‌محور', 'پرداخت امن زرین‌پال', 'پایش ریسک ریل‌تایم'].map((item) => (
              <div key={item} className="flex items-center gap-2 text-sm">
                <ShieldCheck className="h-4 w-4 text-green-400" />
                {item}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="flex items-center justify-center px-4 py-10">
        <form onSubmit={handleSubmit} className="w-full max-w-md persian-card p-6 sm:p-8 rounded-3xl">
          <div className="text-center mb-8">
            <div className="inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-green-500/10 border border-green-500/20 mb-4">
              <UserPlus className="h-7 w-7 text-green-400" />
            </div>
            <h1 className="text-2xl font-black mb-2">ثبت‌نام</h1>
            <p className="text-sm text-muted-foreground">دسترسی به امکانات معاملاتی فین‌دَش</p>
          </div>

          {error && <div className="mb-4 rounded-xl border border-red-500/20 bg-red-500/10 text-red-400 text-sm text-center p-3">{error}</div>}

          <div className="space-y-4">
            <label className="block">
              <span className="block text-sm font-medium mb-1.5">ایمیل</span>
              <div className="relative">
                <Mail className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <input name="email" type="email" dir="ltr" required className="w-full h-12 rounded-2xl border border-input bg-background/70 px-4 pr-10 text-sm outline-none focus:ring-2 focus:ring-green-500/40 focus:border-green-500" />
              </div>
            </label>

            <label className="block">
              <span className="block text-sm font-medium mb-1.5">رمز عبور</span>
              <div className="relative">
                <LockKeyhole className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <input name="password" type="password" dir="ltr" required className="w-full h-12 rounded-2xl border border-input bg-background/70 px-4 pr-10 text-sm outline-none focus:ring-2 focus:ring-green-500/40 focus:border-green-500" />
              </div>
            </label>

            <label className="block">
              <span className="block text-sm font-medium mb-1.5">تأیید رمز عبور</span>
              <div className="relative">
                <LockKeyhole className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <input name="confirm" type="password" dir="ltr" required className="w-full h-12 rounded-2xl border border-input bg-background/70 px-4 pr-10 text-sm outline-none focus:ring-2 focus:ring-green-500/40 focus:border-green-500" />
              </div>
            </label>
          </div>

          <button type="submit" className="btn-persian w-full h-12 rounded-2xl mt-6 flex items-center justify-center gap-2 disabled:opacity-60" disabled={loading}>
            {loading ? "در حال ثبت‌نام..." : "ساخت حساب"}
            {!loading && <ArrowLeft className="h-4 w-4" />}
          </button>

          <div className="text-sm text-center text-muted-foreground mt-6">
            قبلاً ثبت‌نام کرده‌اید؟{' '}
            <Link href="/auth/signin" className="text-green-400 hover:underline font-medium">وارد شوید</Link>
          </div>
        </form>
      </section>
    </main>
  );
}
