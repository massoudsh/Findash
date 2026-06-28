/**
 * TASK-003 — Root Layout with RTL, Vazirmatn font, CurrencyProvider
 */
import type { Metadata } from "next";
import "./globals.css";
import { CurrencyProvider } from "@/context/CurrencyContext";

export const metadata: Metadata = {
  title: "اختاپوس — پلتفرم معاملاتی هوشمند",
  description: "داشبورد مالی جامع با داده‌های بازار ایران",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fa" dir="rtl">
      <head>
        {/* Vazirmatn — best free Persian font for web */}
        <link
          rel="preconnect"
          href="https://fonts.googleapis.com"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body
        className="min-h-screen bg-[#0a0b0f] text-white antialiased"
        style={{ fontFamily: "'Vazirmatn', system-ui, sans-serif" }}
      >
        {/* CurrencyProvider wraps entire app — usdRate hydrated from /api/assets/usd-rate */}
        <CurrencyProvider>
          {children}
        </CurrencyProvider>
      </body>
    </html>
  );
}
