"use client";

import { signIn } from "next-auth/react";
import Link from "next/link";
import { useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { ArrowLeft, LockKeyhole, Mail, ShieldCheck, TrendingUp } from "lucide-react";

function SignInForm() {
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const searchParams = useSearchParams();
  const signupSuccess = searchParams.get("signup") === "success";

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError("");
    setLoading(true);
    const form = e.currentTarget;
    const email = (form.elements.namedItem("email") as HTMLInputElement).value;
    const password = (form.elements.namedItem("password") as HTMLInputElement).value;
    const res = await signIn("credentials", { redirect: false, email, password });
    if (res?.error) {
      setError("ایمیل یا رمز عبور اشتباه است");
      setLoading(false);
    } else {
      window.location.href = "/dashboard";
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
          <h2 className="text-3xl font-black mb-3">به فین دَش خوش آمدید</h2>
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
        <form onSubmit={handleSubmit} className="w-full max-w-md persian-card p-6 sm:p-8 rounded-3xl">
          <div className="text-center mb-8">
            <div className="inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-green-500/10 border border-green-500/20 mb-4">
              <ShieldCheck className="h-7 w-7 text-green-400" />
            </div>
            <h1 className="text-2xl font-black mb-2">ورود به حساب</h1>
            <p className="text-sm text-muted-foreground">برای ادامه وارد داشبورد معاملاتی شوید</p>
          </div>

          {signupSuccess && <div className="mb-4 rounded-xl border border-green-500/20 bg-green-500/10 text-green-400 text-sm text-center p-3">ثبت‌نام موفق! لطفاً وارد شوید.</div>}
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
          </div>

          <button type="submit" className="btn-persian w-full h-12 rounded-2xl mt-6 flex items-center justify-center gap-2 disabled:opacity-60" disabled={loading}>
            {loading ? "در حال ورود..." : "ورود"}
            {!loading && <ArrowLeft className="h-4 w-4" />}
          </button>

          <div className="text-sm text-center text-muted-foreground mt-6">
            حساب ندارید؟{' '}
            <Link href="/auth/signup" className="text-green-400 hover:underline font-medium">ثبت‌نام کنید</Link>
          </div>

          {/* Demo accounts hint */}
          <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-xs text-muted-foreground space-y-2">
            <p className="font-semibold text-foreground/70">حساب‌های آزمایشی:</p>
            {[
              { email: "trader@octopus.trading", pass: "TraderPro2025!" },
              { email: "admin@octopus.trading", pass: "SecureAdmin2025!" },
            ].map((a) => (
              <button
                key={a.email}
                type="button"
                onClick={() => {
                  const form = document.querySelector("form")!;
                  (form.elements.namedItem("email") as HTMLInputElement).value = a.email;
                  (form.elements.namedItem("password") as HTMLInputElement).value = a.pass;
                }}
                className="w-full text-right rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2 hover:border-green-500/20 hover:text-green-400 transition-colors"
                dir="ltr"
              >
                {a.email}
              </button>
            ))}
            <p className="text-[10px] opacity-50">کلیک کن تا فیلدها پر شوند</p>
          </div>
        </form>
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
