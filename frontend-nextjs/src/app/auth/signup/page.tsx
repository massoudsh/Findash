"use client";
import Link from "next/link";
import { useState } from "react";

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
    if (password !== confirm) {
      setError("رمز عبور و تأیید آن یکسان نیستند");
      setLoading(false);
      return;
    }
    // TODO: Replace with real backend call
    if (email === "demo@demo.com") {
      setError("این کاربر قبلاً ثبت‌نام کرده است");
      setLoading(false);
      return;
    }
    // Simulate success
    window.location.href = "/auth/signin?signup=success";
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <form onSubmit={handleSubmit} className="bg-card p-8 rounded-xl shadow-lg w-full max-w-md space-y-6 border border-border">
        <h1 className="text-2xl font-bold mb-4 text-center">ثبت‌نام</h1>
        {error && <div className="text-red-600 text-sm text-center">{error}</div>}
        <div>
          <label htmlFor="email" className="block text-sm font-medium mb-1">ایمیل</label>
          <input name="email" type="email" dir="ltr" required className="mt-1 block w-full border border-input bg-background rounded-md px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary" />
        </div>
        <div>
          <label htmlFor="password" className="block text-sm font-medium mb-1">رمز عبور</label>
          <input name="password" type="password" dir="ltr" required className="mt-1 block w-full border border-input bg-background rounded-md px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary" />
        </div>
        <div>
          <label htmlFor="confirm" className="block text-sm font-medium mb-1">تأیید رمز عبور</label>
          <input name="confirm" type="password" dir="ltr" required className="mt-1 block w-full border border-input bg-background rounded-md px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary" />
        </div>
        <button type="submit" className="w-full bg-primary text-primary-foreground py-2 rounded-md font-semibold hover:bg-primary/90 transition disabled:opacity-50 disabled:cursor-not-allowed" disabled={loading}>
          {loading ? "در حال ثبت‌نام..." : "ثبت‌نام"}
        </button>
        <div className="text-sm text-center text-muted-foreground">
          قبلاً ثبت‌نام کرده‌اید؟{' '}
          <Link href="/auth/signin" className="text-primary hover:underline">وارد شوید</Link>
        </div>
      </form>
    </div>
  );
} 