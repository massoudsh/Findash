'use client';

import Link from 'next/link';
import { useState } from 'react';
import { OverviewDashboard } from '@/app/dashboard/page';
import { ArrowLeft, Info, LogIn, Sparkles } from 'lucide-react';

export default function DemoPage() {
  const [riskValue] = useState(34);

  return (
    <main className="min-h-screen overflow-hidden bg-[#020617] text-slate-100">
      <div className="pointer-events-none fixed inset-0 z-0 bg-[radial-gradient(circle_at_10%_0%,rgba(59,130,246,0.26),transparent_32%),radial-gradient(circle_at_90%_10%,rgba(14,165,233,0.16),transparent_30%),linear-gradient(180deg,rgba(15,23,42,0)_0%,rgba(2,6,23,1)_70%)]" />
      <div className="relative z-10 mx-auto max-w-[1500px] px-3 pb-10 pt-3 sm:px-5 lg:px-7">
        {/* Demo notice bar */}
        <div className="sticky top-0 z-30 -mx-3 border-b border-blue-400/20 bg-blue-950/60 px-3 py-2.5 backdrop-blur-2xl sm:-mx-5 sm:px-5 lg:-mx-7 lg:px-7">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-xs font-bold text-blue-200">
              <Info className="h-4 w-4 shrink-0" />
              این یک نسخه نمایشی با داده‌های آزمایشی است — برای دسترسی به داده واقعی و امکانات کامل ثبت‌نام کنید.
            </div>
            <div className="flex items-center gap-2">
              <Link
                href="/auth/signup"
                className="inline-flex items-center gap-1.5 rounded-full bg-[#3B82F6] px-3.5 py-1.5 text-xs font-black text-white shadow-lg shadow-blue-950/30 transition hover:bg-blue-400"
              >
                <Sparkles className="h-3.5 w-3.5" />
                ثبت‌نام رایگان
              </Link>
              <Link
                href="/auth/signin"
                className="inline-flex items-center gap-1.5 rounded-full border border-white/15 bg-white/[0.05] px-3.5 py-1.5 text-xs font-bold text-slate-200 transition hover:border-blue-400/30 hover:text-blue-300"
              >
                <LogIn className="h-3.5 w-3.5" />
                ورود
              </Link>
            </div>
          </div>
        </div>

        <header className="py-6 sm:py-8">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-blue-400/20 bg-blue-500/10 px-3 py-1.5 text-xs font-bold text-blue-300">
                <Sparkles className="h-3.5 w-3.5" />
                پیش‌نمایش عمومی — بدون نیاز به ورود
              </div>
              <h1 className="text-3xl font-black tracking-tight text-white sm:text-4xl lg:text-5xl">
                فین دَش را قبل از ثبت‌نام امتحان کنید
              </h1>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-400">
                این نمای کلی داشبورد با داده‌های نمونه پر شده تا حس واقعی از تجربه سرمایه‌گذاری، ریسک و پرتفولیو بگیرید.
                برای اتصال به حساب واقعی و بازار زنده وارد شوید یا ثبت‌نام کنید.
              </p>
            </div>
            <Link
              href="/auth/signup"
              className="inline-flex items-center justify-center gap-2 rounded-2xl bg-[#3B82F6] px-5 py-3 text-sm font-black text-white shadow-lg shadow-blue-950/30 transition hover:bg-blue-400"
            >
              شروع رایگان
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </div>
        </header>

        <OverviewDashboard riskValue={riskValue} />

        <div className="mt-8 rounded-[28px] border border-blue-400/20 bg-gradient-to-br from-blue-600/20 via-blue-500/10 to-slate-900 p-6 text-center">
          <h2 className="text-xl font-black text-white">آماده‌اید شروع کنید؟</h2>
          <p className="mt-2 text-sm text-slate-300">
            با یک حساب رایگان، پرتفولیوی واقعی، معاملات و هشدارهای زنده خودتان را بسازید.
          </p>
          <Link
            href="/auth/signup"
            className="mt-4 inline-flex items-center gap-2 rounded-2xl bg-[#3B82F6] px-5 py-3 text-sm font-black text-white shadow-lg shadow-blue-950/30 transition hover:bg-blue-400"
          >
            ثبت‌نام رایگان
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </div>
      </div>
    </main>
  );
}
