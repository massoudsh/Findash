"use client";

/**
 * TASK-004 — CurrencyToggle
 * Small pill button to switch between IRT (تومان) and USD ($)
 */

import { useCurrency } from "@/context/CurrencyContext";

export function CurrencyToggle() {
  const { currency, toggle } = useCurrency();

  return (
    <button
      onClick={toggle}
      className="flex items-center gap-0 rounded-lg overflow-hidden border border-white/10 text-[11px] font-medium shrink-0"
      title="تغییر واحد پولی"
    >
      <span
        className={`px-2.5 py-1 transition-colors ${
          currency === "IRT"
            ? "bg-white/10 text-white"
            : "bg-transparent text-muted-foreground hover:text-white"
        }`}
      >
        ت
      </span>
      <span
        className={`px-2.5 py-1 transition-colors ${
          currency === "USD"
            ? "bg-white/10 text-white"
            : "bg-transparent text-muted-foreground hover:text-white"
        }`}
      >
        $
      </span>
    </button>
  );
}
