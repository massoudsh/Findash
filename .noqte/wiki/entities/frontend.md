# Frontend

> رابط کاربری وب پلتفرم اختاپوس — Next.js 15 با App Router.

## مسئولیت‌ها
- نمایش داشبورد، پورتفولیو، و ابزارهای معاملاتی
- ارتباط ریل‌تایم با backend از طریق WebSocket
- تجسم داده‌های بازار با Recharts و TradingView Charts

## صفحات اصلی
| مسیر | عملکرد |
|------|--------|
| `/dashboard` | داشبورد اصلی + پورتفولیو + My Trades (3 تب) |
| `/realtime` | داده‌های زنده بازار با WebSocket + polling fallback |
| `/news` | اخبار بازار ایران (طلا، ارز، بورس، کریپتو) |
| `/alerts` | هشدار قیمت با localStorage + toast notification |
| `/trades` | مرکز معاملات، سفارش‌ها، تاریخچه |
| `/portfolio` | آنالیز چند دارایی |
| `/trading-bots` | ساخت و مدیریت ربات |
| `/ai-models` | مارکت‌پلیس مدل، آموزش، پیش‌بینی |
| `/risk` | VaR، stress test، متریک‌های ریسک |

## کامپوننت‌های جدید (این مکالمه)
| فایل | عملکرد |
|------|--------|
| `src/lib/hooks/use-market-ws.ts` | WebSocket hook با auto-reconnect و polling fallback |
| `src/lib/hooks/use-price-alerts.ts` | هشدار قیمت در localStorage + trigger callback |
| `src/components/portfolio/trade-tracker.tsx` | ثبت خرید/فروش، محاسبه P&L واقعی |
| `src/components/alerts/alerts-panel.tsx` | پنل ایجاد و مدیریت هشدار قیمت |
| `src/components/data/data-explorer.tsx` | جدول داده‌های بازار با فیلتر |
| `src/app/news/page.tsx` | صفحه اخبار ایران با RSS feed |
| `src/app/alerts/page.tsx` | صفحه مستقل هشدارهای قیمت |

## وابستگی‌ها
- [[entities/backend]] — API calls و WebSocket
- [[concepts/trading-flow]] — flow سفارش‌گذاری

## تکنولوژی
- Next.js 15 (App Router), TypeScript
- Tailwind CSS, Shadcn UI, Radix UI
- React Query (server state), Zustand (client state)
- Recharts, TradingView Charts

## منابع کد
- `MyProjects/Octopus/README.md:559` — ساختار پروژه frontend
- پورت پیش‌فرض: `localhost:3002`
