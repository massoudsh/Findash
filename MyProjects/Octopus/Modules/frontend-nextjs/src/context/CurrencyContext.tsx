"use client";

/**
 * TASK-004 — CurrencyContext
 * Global currency preference: "IRT" (Toman) | "USD"
 * Provides: currency, toggle, format(), convert()
 */

import { createContext, useContext, useState, useCallback, ReactNode } from "react";

export type Currency = "IRT" | "USD";

interface CurrencyContextValue {
  currency: Currency;
  toggle: () => void;
  /** Format a Toman value according to current currency preference */
  format: (tomanValue: number, usdRate?: number) => string;
  /** Convert Toman → USD if current currency is USD */
  convert: (tomanValue: number, usdRate: number) => number;
}

const CurrencyContext = createContext<CurrencyContextValue | null>(null);

export function CurrencyProvider({
  children,
  usdRate = 0,
}: {
  children: ReactNode;
  usdRate?: number;
}) {
  const [currency, setCurrency] = useState<Currency>("IRT");

  const toggle = useCallback(() => {
    setCurrency((c) => (c === "IRT" ? "USD" : "IRT"));
  }, []);

  const convert = useCallback(
    (tomanValue: number, rate: number) => {
      const r = rate || usdRate;
      return r > 0 ? tomanValue / r : 0;
    },
    [usdRate]
  );

  const format = useCallback(
    (tomanValue: number, rate?: number) => {
      if (currency === "USD") {
        const r = rate || usdRate;
        const usd = r > 0 ? tomanValue / r : 0;
        return `$${usd.toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
      }
      // IRT — Toman with Persian-friendly abbreviation
      if (tomanValue >= 1_000_000_000)
        return `${(tomanValue / 1_000_000_000).toFixed(2)} میلیارد ت`;
      if (tomanValue >= 1_000_000)
        return `${(tomanValue / 1_000_000).toFixed(1)} میلیون ت`;
      return new Intl.NumberFormat("fa-IR").format(Math.round(tomanValue)) + " ت";
    },
    [currency, usdRate]
  );

  return (
    <CurrencyContext.Provider value={{ currency, toggle, format, convert }}>
      {children}
    </CurrencyContext.Provider>
  );
}

export function useCurrency() {
  const ctx = useContext(CurrencyContext);
  if (!ctx) throw new Error("useCurrency must be used inside <CurrencyProvider>");
  return ctx;
}
