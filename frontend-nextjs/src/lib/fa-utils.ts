// Persian / Farsi utility functions
// Convention: store English digits in DB, convert at display layer only.

const FA_DIGITS = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'];

/** Convert English digits to Persian: 1234 → ۱۲۳۴ */
export const toPersian = (n: number | string): string =>
  String(n).replace(/[0-9]/g, (d) => FA_DIGITS[+d]);

/** Convert Persian/Arabic digits to English: ۱۲۳۴ → 1234 */
export const toEnglish = (s: string): string =>
  s
    .replace(/[۰-۹]/g, (d) => String(FA_DIGITS.indexOf(d)))
    .replace(/[٠-٩]/g, (d) => String('٠١٢٣٤٥٦٧٨٩'.indexOf(d)));

/** Format a number as Iranian Toman: 1234567 → ۱٬۲۳۴٬۵۶۷ تومان */
export const formatToman = (amount: number): string =>
  toPersian(amount.toLocaleString('en-US')) + ' تومان';

/** Short Toman format for large numbers */
export const formatTomanShort = (amount: number): string => {
  if (amount >= 1_000_000_000)
    return toPersian((amount / 1_000_000_000).toFixed(1)) + ' میلیارد تومان';
  if (amount >= 1_000_000)
    return toPersian((amount / 1_000_000).toFixed(1)) + ' میلیون تومان';
  return formatToman(amount);
};

// ── Jalali (Shamsi) calendar ─────────────────────────────────────────────────
// Minimal implementation — no external dependency needed.

function gregorianToJalali(
  gy: number,
  gm: number,
  gd: number
): [number, number, number] {
  const g_d_no = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
  const j_d_no = [0, 31, 62, 93, 124, 155, 186, 216, 246, 276, 306, 336];
  let jy: number, jm: number, jd: number;

  let gy2 = gm > 2 ? gy + 1 : gy;
  let g_day_no =
    365 * (gy - 1) +
    Math.floor((gy2 + 3) / 4) -
    Math.floor((gy2 + 99) / 100) +
    Math.floor((gy2 + 399) / 400);

  for (let i = 0; i < gm - 1; i++) g_day_no += g_d_no[i];
  if (gm > 2 && ((gy % 4 === 0 && gy % 100 !== 0) || gy % 400 === 0)) g_day_no++;
  g_day_no += gd - 1;

  let j_day_no = g_day_no - 79;
  const j_np = Math.floor(j_day_no / 12053);
  j_day_no %= 12053;
  jy = 979 + 33 * j_np + 4 * Math.floor(j_day_no / 1461);
  j_day_no %= 1461;

  if (j_day_no >= 366) {
    jy += Math.floor((j_day_no - 1) / 365);
    j_day_no = (j_day_no - 1) % 365;
  }

  let i = 0;
  for (; i < 11 && j_day_no >= j_d_no[i + 1]; i++) {}
  jm = i + 1;
  jd = j_day_no - j_d_no[i] + 1;

  return [jy, jm, jd];
}

/** Convert a Gregorian Date to Jalali display string: "۱۴۰۵/۰۳/۱۰" */
export function formatJalali(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  const [jy, jm, jd] = gregorianToJalali(
    d.getFullYear(),
    d.getMonth() + 1,
    d.getDate()
  );
  const pad = (n: number) => toPersian(String(n).padStart(2, '0'));
  return `${toPersian(jy)}/${pad(jm)}/${pad(jd)}`;
}

const JALALI_MONTHS = [
  'فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور',
  'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند',
];

/** Jalali with month name: "۱۰ خرداد ۱۴۰۵" */
export function formatJalaliLong(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  const [jy, jm, jd] = gregorianToJalali(d.getFullYear(), d.getMonth() + 1, d.getDate());
  return `${toPersian(jd)} ${JALALI_MONTHS[jm - 1]} ${toPersian(jy)}`;
}

/** Relative time in Persian: "۳ دقیقه پیش" */
export function relativeFa(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return 'همین الان';
  if (diff < 3600) return `${toPersian(Math.floor(diff / 60))} دقیقه پیش`;
  if (diff < 86400) return `${toPersian(Math.floor(diff / 3600))} ساعت پیش`;
  if (diff < 2592000) return `${toPersian(Math.floor(diff / 86400))} روز پیش`;
  return formatJalali(d);
}

/** Format percentage in Persian: 12.5 → "۱۲٫۵٪" */
export const formatPercentFa = (value: number, decimals = 2): string =>
  toPersian(Math.abs(value).toFixed(decimals)) + '٪';
