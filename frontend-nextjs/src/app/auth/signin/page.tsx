"use client";
import { signIn } from "next-auth/react";
import Link from "next/link";
import { useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";

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
    const res = await signIn("credentials", {
      redirect: false,
      email,
      password,
    });
    if (res?.error) {
      setError("ایمیل یا رمز عبور اشتباه است");
      setLoading(false);
    } else {
      window.location.href = "/dashboard";
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <form onSubmit={handleSubmit} className="bg-card p-8 rounded-xl shadow-lg w-full max-w-md space-y-6 border border-border">
        <h1 className="text-2xl font-bold mb-4 text-center">ورود به حساب</h1>
        {signupSuccess && <div className="text-green-600 text-sm text-center">ثبت‌نام موفق! لطفاً وارد شوید.</div>}
        {error && <div className="text-red-600 text-sm text-center">{error}</div>}
        <div>
          <label htmlFor="email" className="block text-sm font-medium mb-1">ایمیل</label>
          <input name="email" type="email" dir="ltr" required className="mt-1 block w-full border border-input bg-background rounded-md px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary" />
        </div>
        <div>
          <label htmlFor="password" className="block text-sm font-medium mb-1">رمز عبور</label>
          <input name="password" type="password" dir="ltr" required className="mt-1 block w-full border border-input bg-background rounded-md px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary" />
        </div>
        <button type="submit" className="w-full bg-primary text-primary-foreground py-2 rounded-md font-semibold hover:bg-primary/90 transition disabled:opacity-50 disabled:cursor-not-allowed" disabled={loading}>
          {loading ? "در حال ورود..." : "ورود"}
        </button>
        <div className="text-sm text-center text-muted-foreground">
          حساب ندارید؟{' '}
          <Link href="/auth/signup" className="text-primary hover:underline">ثبت‌نام کنید</Link>
        </div>
      </form>
    </div>
  );
}

export default function SignInPage() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center">در حال بارگذاری…</div>}>
      <SignInForm />
    </Suspense>
  );
} 