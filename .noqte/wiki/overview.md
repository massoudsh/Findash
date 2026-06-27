# Overview — Octopus Trading Platform

> پلتفرم معاملاتی هوشمند با هوش مصنوعی، تحلیل ریل‌تایم، و مدیریت ریسک پیشرفته.

## وضعیت پروژه
- **نوع:** اپلیکیشن وب (داشبورد مالی همه‌کاره)
- **وضعیت:** غیرفعال
- **نسخه فعلی:** v0.4.0

## خلاصه معماری
پروژه از سه لایه اصلی تشکیل شده:

1. **Frontend** — Next.js 15 (TypeScript) روی پورت `3002`
2. **Backend** — FastAPI (Python 3.10+) روی پورت `8000`
3. **Data Layer** — PostgreSQL + TimescaleDB، Redis Cache، Kafka Streaming

لایه‌ی هوش مصنوعی شامل `IntelligenceOrchestrator` است که ۱۱ AI Agent را هماهنگ می‌کند. داده‌های بازار از طریق Kafka دریافت، در Redis کش، و در TimescaleDB ذخیره می‌شوند. وظایف سنگین از طریق Celery Workers پردازش می‌شوند.

## قابلیت‌های اصلی
- معامله‌گری چند دارایی (سهام، آپشن، کریپتو)
- داده‌های بازار ریل‌تایم با WebSocket
- مدل‌های ML برای پیش‌بینی قیمت و تحلیل سنتیمنت
- ربات‌های معاملاتی خودکار با backtesting
- مدیریت ریسک (VaR، stress testing)
- داشبورد مانیتورینگ با Prometheus + Grafana

## تکنولوژی‌ها
| لایه | فناوری |
|------|--------|
| Frontend | Next.js 15, TypeScript, Tailwind CSS, Shadcn UI, Recharts |
| Backend | FastAPI, Python 3.10+, Celery, WebSockets |
| AI/ML | PyTorch, TensorFlow, scikit-learn |
| Database | PostgreSQL, TimescaleDB, Redis |
| Streaming | Kafka (Producer/Consumer) |
| Monitoring | Prometheus (9090), Grafana (3001), Flower (5555) |
| Infrastructure | Docker |

## منابع
- `MyProjects/Octopus/README.md` — مستندات اصلی
- `MyProjects/Octopus/docs/orchestrator-architecture-detailed.md` — معماری کامل Orchestrator
