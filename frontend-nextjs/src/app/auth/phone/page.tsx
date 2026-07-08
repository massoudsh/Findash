'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Phone, ArrowLeft, Loader2 } from 'lucide-react';
import Link from 'next/link';

export default function PhoneAuthPage() {
  const router = useRouter();
  const [phone, setPhone] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const res = await fetch('/api/proxy/auth/send-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || 'خطا در ارسال کد');
        return;
      }

      // Save phone in sessionStorage for OTP page
      sessionStorage.setItem('otp_phone', phone);
      router.push('/auth/otp');
    } catch {
      setError('خطا در اتصال به سرور. اینترنت را بررسی کنید.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#030712] p-4">
      <div className="w-full max-w-sm">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-green-500/10 border border-green-500/20">
            <Phone className="h-8 w-8 text-green-400" />
          </div>
          <h1 className="text-2xl font-black text-white">ورود با موبایل</h1>
          <p className="mt-2 text-sm text-slate-400">
            کد تأیید به شماره شما پیامک می‌شود
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-2">
            <Label htmlFor="phone" className="text-sm font-bold text-slate-300">
              شماره موبایل
            </Label>
            <Input
              id="phone"
              type="tel"
              dir="ltr"
              placeholder="09121234567"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              className="h-12 rounded-2xl border-white/10 bg-white/[0.04] text-center text-lg tracking-widest text-white placeholder:text-slate-600 focus:border-green-500/40"
              maxLength={11}
              required
            />
          </div>

          {error && (
            <p className="rounded-xl bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-400 text-center">
              {error}
            </p>
          )}

          <Button
            type="submit"
            disabled={loading || phone.length < 10}
            className="w-full h-12 rounded-2xl bg-green-500 text-black font-black hover:bg-green-400 disabled:opacity-50"
          >
            {loading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <>
                ارسال کد تأیید
                <ArrowLeft className="mr-2 h-4 w-4" />
              </>
            )}
          </Button>
        </form>

        {/* Divider */}
        <div className="mt-6 flex items-center gap-3">
          <div className="flex-1 h-px bg-white/10" />
          <span className="text-xs text-slate-500">یا</span>
          <div className="flex-1 h-px bg-white/10" />
        </div>

        {/* Back to email login */}
        <div className="mt-4 text-center">
          <Link
            href="/auth/signin"
            className="text-sm text-slate-400 hover:text-white transition"
          >
            ورود با ایمیل و رمز عبور
          </Link>
        </div>
      </div>
    </div>
  );
}
