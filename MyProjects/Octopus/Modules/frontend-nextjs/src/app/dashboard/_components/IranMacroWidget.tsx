"use client";

/**
 * TASK-002b — IranMacroWidget
 * نمایش شاخص‌های کلان اقتصاد ایران:
 * نرخ تورم، رشد اقتصادی، نرخ ارز رسمی vs آزاد، و قیمت نفت
 *
 * Data sources:
 * - تورم + رشد: بانک مرکزی (CBI) / scrape از tgju.org
 * - نرخ ارز رسمی: بانک مرکزی
 * - نفت: tgju.org / tradingeconomics
 *
 * برای نمایش، مقادیر به‌صورت static seed هستند تا بدون API کار کند.
 * در production: از /api/macro endpoint بخوان (بعداً اضافه می‌شود)
 */

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface MacroItem {
  label: string;
  value: string;
  change?: number;   // positive = good, negative = bad
  note?: string;
  icon: string;
}

// Static seed — در production از API بخوان
const MACRO_ITEMS: MacroItem[] = [
  {
    label:  "نرخ تورم سالانه",
    value:  "۳۳٪",
    change: -2.1,   // بهبود نسبت به ماه قبل
    note:   "بهمن ۱۴۰۳",
    icon:   "📈",
  },
  {
    label:  "رشد اقتصادی",
    value:  "۴.۲٪",
    change: +0.3,
    note:   "سال ۱۴۰۳",
    icon:   "🏭",
  },
  {
    label:  "دلار رسمی (بانک مرکزی)",
    value:  "۵۸,۰۰۰ ت",
    change: 0,
    icon:   "🏦",
  },
  {
    label:  "نفت برنت",
    value:  "$۷۲",
    change: -1.4,
    note:   "هر بشکه",
    icon:   "🛢️",
  },
  {
    label:  "شاخص بورس (TEDPIX)",
    value:  "۲.۱ میلیون",
    change: +0.8,
    note:   "آخرین معامله",
    icon:   "📊",
  },
  {
    label:  "نرخ بهره بانکی",
    value:  "۲۳٪",
    change: 0,
    note:   "سپرده ۱ ساله",
    icon:   "🏧",
  },
];

export function IranMacroWidget() {
  return (
    <Card className="bg-[#0f1117] border border-white/5" dir="rtl">
      <CardHeader className="pb-2 pt-4 px-4">
        <div className="flex items-center gap-2">
          <span className="text-base">🇮🇷</span>
          <h3 className="text-sm font-semibold text-white">شاخص‌های کلان ایران</h3>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        <div className="grid grid-cols-2 gap-2">
          {MACRO_ITEMS.map((item) => (
            <div
              key={item.label}
              className="flex flex-col gap-0.5 p-2.5 rounded-lg bg-white/[0.03] border border-white/5"
            >
              <div className="flex items-center gap-1.5">
                <span className="text-sm leading-none">{item.icon}</span>
                <span className="text-[11px] text-muted-foreground truncate">{item.label}</span>
              </div>
              <div className="flex items-center justify-between mt-1">
                <span className="text-sm font-semibold text-white tabular-nums">
                  {item.value}
                </span>
                {item.change !== undefined && item.change !== 0 && (
                  <span
                    className={`flex items-center gap-0.5 text-[11px] font-medium ${
                      item.change > 0 ? "text-emerald-400" : "text-red-400"
                    }`}
                  >
                    {item.change > 0
                      ? <TrendingUp className="w-3 h-3" />
                      : <TrendingDown className="w-3 h-3" />
                    }
                    {Math.abs(item.change)}٪
                  </span>
                )}
                {item.change === 0 && (
                  <Minus className="w-3 h-3 text-muted-foreground" />
                )}
              </div>
              {item.note && (
                <span className="text-[10px] text-muted-foreground/60">{item.note}</span>
              )}
            </div>
          ))}
        </div>
        <p className="text-[10px] text-muted-foreground/40 mt-3 text-center">
          منابع: بانک مرکزی · tgju.org · بورس اوراق بهادار
        </p>
      </CardContent>
    </Card>
  );
}
