/**
 * TASK-003 — Locale utilities
 * Jalali (Persian) date conversion, RTL helpers, Persian number formatting
 *
 * Note: Full Jalali requires jalaali-js (npm i jalaali-js).
 * This file includes a self-contained lightweight conversion for the
 * most common use case (date display) so there are no extra deps.
 */

// ─── Jalali conversion ───────────────────────────────────────────────────────

/** Convert a Gregorian Date to Jalali {year, month, day} */
export function toJalali(date: Date): { year: number; month: number; day: number } {
  const gy = date.getFullYear();
  const gm = date.getMonth() + 1;
  const gd = date.getDate();

  const g_d_no =
    365 * gy +
    Math.floor((gy + 3) / 4) -
    Math.floor((gy + 99) / 100) +
    Math.floor((gy + 399) / 400);
  const g_d_no2 = [0, 31, 28 + (gy % 4 === 0 && (gy % 100 !== 0 || gy % 400 === 0) ? 1 : 0), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  let sum = 0;
  for (let i = 1; i < gm; i++) sum += g_d_no2[i];
  const gg = g_d_no + sum + gd - 1;

  const j_d_no = gg - 79;
  const j_np  = Math.floor(j_d_no / 12053);
  const rem   = j_d_no % 12053;
  const jy    = 979 + 33 * j_np + 4 * Math.floor(rem / 1461);
  const rem2  = rem % 1461;
  let jm      = 0;
  let jd      = 0;
  if (rem2 >= 366) {
    const rem3 = rem2 - 1;
    const tmp  = Math.floor(rem3 / 365);
    const jyy  = jy + tmp;
    const rem4 = rem3 % 365;
    const jml  = [31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29];
    let s = 0;
    for (let i = 0; i < 12; i++) {
      if (rem4 < s + jml[i]) { jm = i + 1; jd = rem4 - s + 1; break; }
      s += jml[i];
    }
    return { year: jyy, month: jm, day: jd };
  } else {
    const jml = [31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29];
    let s = 0;
    for (let i = 0; i < 12; i++) {
      if (rem2 < s + jml[i]) { jm = i + 1; jd = rem2 - s + 1; break; }
      s += jml[i];
    }
    return { year: jy, month: jm, day: jd };
  }
}

const JALALI_MONTHS = [
  "فروردین","اردیبهشت","خرداد","تیر","مرداد","شهریور",
  "مهر","آبان","آذر","دی","بهمن","اسفند",
];

/** Format Date as "۱۴ خرداد ۱۴۰۳" */
export function formatJalali(date: Date | string, short = false): string {
  const d = typeof date === "string" ? new Date(date) : date;
  if (isNaN(d.getTime())) return "-";
  const { year, month, day } = toJalali(d);
  const monthName = JALALI_MONTHS[month - 1];
  if (short) return `${toPersianDigits(day)} ${monthName.slice(0, 3)}`;
  return `${toPersianDigits(day)} ${monthName} ${toPersianDigits(year)}`;
}

/** Format as short date "۱۴۰۳/۰۳/۱۴" */
export function formatJalaliShort(date: Date | string): string {
  const d = typeof date === "string" ? new Date(date) : date;
  if (isNaN(d.getTime())) return "-";
  const { year, month, day } = toJalali(d);
  return `${toPersianDigits(year)}/${toPersianDigits(String(month).padStart(2, "0"))}/${toPersianDigits(String(day).padStart(2, "0"))}`;
}

// ─── Persian digits ──────────────────────────────────────────────────────────

const FA_DIGITS = ["۰","۱","۲","۳","۴","۵","۶","۷","۸","۹"];

export function toPersianDigits(value: string | number): string {
  return String(value).replace(/\d/g, (d) => FA_DIGITS[parseInt(d)]);
}

/** Optionally convert number string to Persian digits based on locale preference */
export function localizeNumber(value: string | number, persian = true): string {
  return persian ? toPersianDigits(value) : String(value);
}

// ─── RTL helpers ─────────────────────────────────────────────────────────────

/** Direction for a given locale */
export const DIR = "rtl" as const;

/** CSS class for RTL text alignment */
export const RTL_CLASS = "text-right font-[Vazirmatn,_system-ui,_sans-serif]";
