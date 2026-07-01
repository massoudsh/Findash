/**
 * تبدیل تاریخ میلادی به شمسی
 * Pure JS — no external dependency
 */

const FA_DIGITS = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'];

export function toFarsiDigits(n: number | string): string {
  return String(n).replace(/\d/g, (d) => FA_DIGITS[+d]);
}

/** تبدیل تاریخ میلادی به شمسی — الگوریتم خلیلی */
export function toJalali(date: Date | string | number): { year: number; month: number; day: number } {
  const d = new Date(date);
  const gy = d.getFullYear();
  const gm = d.getMonth() + 1;
  const gd = d.getDate();

  const g_d_no = 365 * gy + Math.floor((gy + 3) / 4) - Math.floor((gy + 99) / 100) + Math.floor((gy + 399) / 400);
  let g_d_no2 = g_d_no;

  const g_days = [0, 31, 28 + (gy % 4 === 0 && (gy % 100 !== 0 || gy % 400 === 0) ? 1 : 0), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  for (let i = 1; i < gm; i++) g_d_no2 += g_days[i];
  g_d_no2 += gd;

  const j_d_no = g_d_no2 - 79;
  const j_np = Math.floor(j_d_no / 12053);
  let j_d_no3 = j_d_no % 12053;
  let jy = 979 + 33 * j_np + 4 * Math.floor(j_d_no3 / 1461);
  j_d_no3 %= 1461;

  if (j_d_no3 >= 366) {
    jy += Math.floor((j_d_no3 - 1) / 365);
    j_d_no3 = (j_d_no3 - 1) % 365;
  }

  const j_days = [0, 31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29];
  let jm = 0;
  for (jm = 1; jm <= 11 && j_days[jm] <= j_d_no3; jm++) j_d_no3 -= j_days[jm];
  const jd = j_d_no3 + 1;

  return { year: jy, month: jm, day: jd };
}

const MONTH_NAMES = [
  '', 'فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور',
  'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند',
];

/** فرمت کامل: ۱۴۰۵/۰۴/۱۰ */
export function formatJalali(date: Date | string | number): string {
  const { year, month, day } = toJalali(date);
  return `${toFarsiDigits(year)}/${toFarsiDigits(String(month).padStart(2, '0'))}/${toFarsiDigits(String(day).padStart(2, '0'))}`;
}

/** فرمت خواناتر: ۱۰ تیر ۱۴۰۵ */
export function formatJalaliLong(date: Date | string | number): string {
  const { year, month, day } = toJalali(date);
  return `${toFarsiDigits(day)} ${MONTH_NAMES[month]} ${toFarsiDigits(year)}`;
}

/** زمان نسبی فارسی */
export function formatRelative(date: Date | string | number): string {
  const d = new Date(date);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffSec < 60) return 'همین الان';
  if (diffMin < 60) return `${toFarsiDigits(diffMin)} دقیقه پیش`;
  if (diffHour < 24) return `${toFarsiDigits(diffHour)} ساعت پیش`;
  if (diffDay < 7) return `${toFarsiDigits(diffDay)} روز پیش`;
  return formatJalali(date);
}

/** فرمت ساعت فارسی: ۱۴:۳۵ */
export function formatTime(date: Date | string | number): string {
  const d = new Date(date);
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  return toFarsiDigits(`${h}:${m}`);
}
