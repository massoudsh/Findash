'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { signIn } from 'next-auth/react';
import { Button } from '@/components/ui/button';
import { MessageSquare, Loader2, RotateCcw } from 'lucide-react';
import Link from 'next/link';

const OTP_LENGTH = 6;
const RESEND_COUNTDOWN = 120; // 2 minutes

export default function OTPPage() {
  const router = useRouter();
  const [digits, setDigits] = useState<string[]>(Array(OTP_LENGTH).fill(''));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [countdown, setCountdown] = useState(RESEND_COUNTDOWN);
  const [resending, setResending] = useState(false);
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);
  const [phone, setPhone] = useState('');

  useEffect(() => {
    const savedPhone = sessionStorage.getItem('otp_phone') || '';
    setPhone(savedPhone);
    if (!savedPhone) router.replace('/auth/phone');

    // Focus first input
    inputRefs.current[0]?.focus();
  }, [router]);

  // Countdown timer
  useEffect(() => {
    if (countdown <= 0) return;
    const id = setInterval(() => setCountdown((c) => c - 1), 1000);
    return () => clearInterval(id);
  }, [countdown]);

  const formatCountdown = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  };

  const handleDigitChange = (index: number, value: string) => {
    // Handle paste
    if (value.length > 1) {
      const digits = value.replace(/\D/g, '').slice(0, OTP_LENGTH).split('');
      const newDigits = [...Array(OTP_LENGTH).fill('')];
      digits.forEach((d, i) => { newDigits[i] = d; });
      setDigits(newDigits);
      inputRefs.current[Math.min(digits.length, OTP_LENGTH - 1)]?.focus();
      return;
    }

    const cleaned = value.replace(/\D/g, '').slice(0, 1);
    const newDigits = [...digits];
    newDigits[index] = cleaned;
    setDigits(newDigits);

    // Auto-advance
    if (cleaned && index < OTP_LENGTH - 1) {
      inputRefs.current[index + 1]?.focus();
    }
  };

  const handleKeyDown = (index: number, e: React.KeyboardEvent) => {
    if (e.key === 'Backspace' && !digits[index] && index > 0) {
      inputRefs.current[index - 1]?.focus();
    }
  };

  const code = digits.join('');
  const isComplete = code.length === OTP_LENGTH;

  const handleVerify = async () => {
    if (!isComplete || !phone) return;
    setError('');
    setLoading(true);

    try {
      const res = await fetch('/api/proxy/auth/verify-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone, code }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || 'کد اشتباه است');
        setDigits(Array(OTP_LENGTH).fill(''));
        inputRefs.current[0]?.focus();
        return;
      }

      // Sign in with NextAuth using the token
      sessionStorage.removeItem('otp_phone');
      await signIn('credentials', {
        phone,
        otp_token: data.access_token,
        redirect: true,
        callbackUrl: '/dashboard',
      });
    } catch {
      setError('خطا در اتصال به سرور');
    } finally {
      setLoading(false);
    }
  };

  const handleResend = async () => {
    if (countdown > 0 || !phone) return;
    setResending(true);
    setError('');

    try {
      const res = await fetch('/api/proxy/auth/send-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone }),
      });

      if (res.ok) {
        setCountdown(RESEND_COUNTDOWN);
        setDigits(Array(OTP_LENGTH).fill(''));
        inputRefs.current[0]?.focus();
      } else {
        const data = await res.json();
        setError(data.detail || 'خطا در ارسال مجدد');
      }
    } catch {
      setError('خطا در اتصال به سرور');
    } finally {
      setResending(false);
    }
  };

  // Masked phone display: 0912***4567
  const maskedPhone = phone
    ? phone.slice(0, 4) + '***' + phone.slice(-4)
    : '';

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#030712] p-4">
      <div className="w-full max-w-sm">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-500/10 border border-blue-500/20">
            <MessageSquare className="h-8 w-8 text-blue-400" />
          </div>
          <h1 className="text-2xl font-black text-white">کد تأیید</h1>
          {maskedPhone && (
            <p className="mt-2 text-sm text-slate-400" dir="ltr">
              کد ارسال‌شده به <span className="font-bold text-white">{maskedPhone}</span>
            </p>
          )}
        </div>

        {/* OTP input grid */}
        <div className="flex justify-center gap-3 mb-6" dir="ltr">
          {Array(OTP_LENGTH).fill(null).map((_, i) => (
            <input
              key={i}
              ref={(el) => { inputRefs.current[i] = el; }}
              type="text"
              inputMode="numeric"
              maxLength={6}
              value={digits[i]}
              onChange={(e) => handleDigitChange(i, e.target.value)}
              onKeyDown={(e) => handleKeyDown(i, e)}
              className={`
                w-12 h-14 rounded-2xl border text-center text-2xl font-black text-white
                bg-white/[0.04] outline-none transition-all duration-150
                ${digits[i] ? 'border-green-500/50 bg-green-500/[0.06]' : 'border-white/10'}
                focus:border-blue-400/50 focus:ring-1 focus:ring-blue-400/20
              `}
            />
          ))}
        </div>

        {error && (
          <p className="mb-4 rounded-xl bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-400 text-center">
            {error}
          </p>
        )}

        {/* Verify button */}
        <Button
          onClick={handleVerify}
          disabled={!isComplete || loading}
          className="w-full h-12 rounded-2xl bg-green-500 text-black font-black hover:bg-green-400 disabled:opacity-50"
        >
          {loading ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            'تأیید و ورود'
          )}
        </Button>

        {/* Resend */}
        <div className="mt-5 text-center">
          {countdown > 0 ? (
            <p className="text-sm text-slate-400" dir="ltr">
              ارسال مجدد تا <span className="font-bold text-white">{formatCountdown(countdown)}</span>
            </p>
          ) : (
            <button
              onClick={handleResend}
              disabled={resending}
              className="flex items-center gap-2 mx-auto text-sm text-blue-400 hover:text-blue-300 transition"
            >
              {resending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RotateCcw className="h-4 w-4" />
              )}
              ارسال مجدد کد
            </button>
          )}
        </div>

        {/* Change phone */}
        <div className="mt-4 text-center">
          <Link href="/auth/phone" className="text-xs text-slate-500 hover:text-slate-400 transition">
            تغییر شماره موبایل
          </Link>
        </div>
      </div>
    </div>
  );
}
