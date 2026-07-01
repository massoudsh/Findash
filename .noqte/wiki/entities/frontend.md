# Frontend

> رابط کاربری وب پلتفرم اختاپوس — Next.js 15 با App Router.

## مسئولیت‌ها
- نمایش داشبورد، پورتفولیو، و ابزارهای معاملاتی
- ارتباط ریل‌تایم با backend از طریق WebSocket
- تجسم داده‌های بازار با Recharts و TradingView Charts

## صفحات اصلی
| مسیر | عملکرد |
|------|--------|
| `/` | صفحه اصلی فارسی — phone mockup، market cards، feature cards، CTA |
| `/dashboard` | داشبورد اصلی — 5 تب یکپارچه (نمای کلی / پرتفولیو / بازار / معاملات / تحلیل) |
| `/auth/signin` | ورود — two-column layout، visual preview panel |
| `/auth/signup` | ثبت‌نام — two-column layout، feature list |
| `/news` | اخبار بازار ایران (طلا، ارز، بورس، کریپتو) |
| `/alerts` | هشدار قیمت با localStorage + toast notification |
| `/payment/checkout` | انتخاب پلن و شروع پرداخت زرین‌پال |
| `/payment/callback/zarinpal` | bridge برگشت از زرین‌پال به backend verify |
| `/payment/success` | نتیجه پرداخت موفق |
| `/payment/failed` | نتیجه پرداخت ناموفق |

## کامپوننت‌های اصلی
| فایل | عملکرد |
|------|--------|
| `src/components/dashboard/risk-gauge.tsx` | گیج ریسک SVG semicircle، ۴ سطح رنگی، live mode با drift |
| `src/components/dashboard/credit-score.tsx` | امتیاز اعتباری ۳۰۰-۸۵۰، animated counter، progress bars |
| `src/components/portfolio/trade-tracker.tsx` | ثبت خرید/فروش، محاسبه P&L واقعی |
| `src/lib/hooks/use-market-ws.ts` | WebSocket hook با auto-reconnect و polling fallback |
| `src/lib/hooks/use-price-alerts.ts` | هشدار قیمت در localStorage + trigger callback |
| `src/components/alerts/alerts-panel.tsx` | پنل ایجاد و مدیریت هشدار قیمت |
| `src/app/news/page.tsx` | صفحه اخبار ایران با RSS feed |

## طراحی بصری فعلی
- رنگ اصلی: `#22C55E` (سبز) — CSS var `--primary: 142 71% 45%`
- فونت: IRANYekanX با fallback به Vazirmatn
- CSS classes فارسی: `.persian-card`, `.persian-border`, `.persian-pattern-bg`, `.btn-persian`, `.persian-badge`
- داشبورد: 5 تب یکپارچه (overview/portfolio/market/trades/analytics) — تب state در URL
- موبایل: bottom navigation bar با 5 آیتم، `pb-24 lg:pb-6` برای clearance

## وابستگی‌ها
- [[entities/backend]] — API calls و WebSocket
- [[concepts/trading-flow]] — flow سفارش‌گذاری

## تکنولوژی
- Next.js 15 (App Router), TypeScript
- Tailwind CSS, Shadcn UI, Radix UI
- React Query (server state), Zustand (client state)
- Recharts, TradingView Charts

## منابع کد
- `frontend-nextjs/src/app/payment/checkout/page.tsx` — صفحه انتخاب پلن و پرداخت
- `frontend-nextjs/src/app/payment/callback/zarinpal/page.tsx` — callback bridge زرین‌پال
- `frontend-nextjs/src/app/payment/success/page.tsx` — صفحه موفقیت پرداخت
- `frontend-nextjs/src/app/payment/failed/page.tsx` — صفحه شکست پرداخت
- `frontend-nextjs/src/app/api/payment/zarinpal/create/route.ts` — API proxy پرداخت
- پورت پیش‌فرض: `localhost:3003`
