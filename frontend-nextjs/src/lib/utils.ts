import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export type CurrencyUnit = 'IRT' | 'IRR' | 'USD';

export function formatCurrency(amount: number, unit: CurrencyUnit = 'USD'): string {
  if (unit === 'IRT') {
    const formatted = new Intl.NumberFormat('fa-IR', { maximumFractionDigits: 0 }).format(amount);
    return `${formatted} تومان`;
  }
  if (unit === 'IRR') {
    const formatted = new Intl.NumberFormat('fa-IR', { maximumFractionDigits: 0 }).format(amount);
    return `${formatted} ریال`;
  }
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(amount);
}

export function formatTomanValue(amount: number, unit: 'IRT' | 'IRR'): string {
  return formatCurrency(unit === 'IRR' ? amount * 10 : amount, unit);
}

export function safeCallbackPath(value: string | null, origin: string): string {
  try {
    const target = new URL(value || '/dashboard', origin);
    if (target.origin === origin && !target.pathname.startsWith('//')) return target.pathname + target.search;
  } catch {}
  return '/dashboard';
}

export function formatPercentage(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value / 100)
}

export function formatDate(
  date: Date | string,
  format: 'long' | 'short' = 'long'
): string {
  const options: Intl.DateTimeFormatOptions =
    format === 'long'
      ? {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        }
      : {
          month: 'short',
          day: 'numeric',
        };

  return new Intl.DateTimeFormat('en-US', options).format(new Date(date));
}