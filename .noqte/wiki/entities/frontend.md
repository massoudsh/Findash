# Frontend

> رابط کاربری وب پلتفرم اختاپوس — Next.js 15 با App Router.

## مسئولیت‌ها
- نمایش داشبورد، پورتفولیو، و ابزارهای معاملاتی
- ارتباط ریل‌تایم با backend از طریق WebSocket
- تجسم داده‌های بازار با Recharts و TradingView Charts

## صفحات اصلی
| مسیر | عملکرد |
|------|--------|
| `/` | داشبورد اصلی + پورتفولیو |
| `/realtime` | داده‌های زنده بازار، orderbook، سنتیمنت |
| `/trades` | مرکز معاملات، سفارش‌ها، تاریخچه |
| `/portfolio` | آنالیز چند دارایی |
| `/trading-bots` | ساخت و مدیریت ربات |
| `/ai-models` | مارکت‌پلیس مدل، آموزش، پیش‌بینی |
| `/risk` | VaR، stress test، متریک‌های ریسک |
| `/backtesting` | آزمون استراتژی تاریخی |

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
