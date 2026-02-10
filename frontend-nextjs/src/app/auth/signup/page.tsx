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
      setError("Passwords do not match");
      setLoading(false);
      return;
    }
    // TODO: Replace with real backend call
    if (email === "demo@demo.com") {
      setError("User already exists");
      setLoading(false);
      return;
    }
    // Simulate success
    window.location.href = "/auth/signin?signup=success";
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <form onSubmit={handleSubmit} className="bg-white p-8 rounded shadow-md w-full max-w-md space-y-6">
        <h1 className="text-2xl font-bold mb-4">Sign up</h1>
        {error && <div className="text-red-600">{error}</div>}
        <div>
          <label htmlFor="email" className="block text-sm font-medium text-gray-700">Email</label>
          <input name="email" type="email" required className="mt-1 block w-full border border-input bg-background rounded-md px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary" />
        </div>
        <div>
          <label htmlFor="password" className="block text-sm font-medium text-gray-700">Password</label>
          <input name="password" type="password" required className="mt-1 block w-full border border-input bg-background rounded-md px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary" />
        </div>
        <div>
          <label htmlFor="confirm" className="block text-sm font-medium text-gray-700">Confirm Password</label>
          <input name="confirm" type="password" required className="mt-1 block w-full border border-input bg-background rounded-md px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary" />
        </div>
        <button type="submit" className="w-full bg-primary text-white py-2 rounded-md font-semibold hover:bg-primary/90 transition disabled:opacity-50 disabled:cursor-not-allowed" disabled={loading}>{loading ? "Signing up..." : "Sign up"}</button>
        <div className="text-sm text-center">
          Already have an account? <Link href="/auth/signin" className="text-primary hover:underline">Sign in</Link>
        </div>
      </form>
    </div>
  );
} 