# Trading Flow

> flow کامل از ثبت سفارش توسط کاربر تا اجرا در broker.

## مراحل
1. کاربر سفارش را در Frontend (`/trades`) ثبت می‌کند
2. Frontend → API Gateway (احراز هویت + rate limiting)
3. Backend: بررسی موجودی از PostgreSQL
4. Backend → Orchestrator/AI: ارزیابی ریسک (Risk Score)
5. اگر ریسک قابل‌قبول: ارسال به Trading Broker خارجی
6. تأیید سفارش → آپدیت پورتفولیو در PostgreSQL
7. Frontend دریافت نتیجه (WebSocket یا Response)

## اجزای درگیر
- [[entities/frontend]] — UI ثبت سفارش
- [[entities/backend]] — validation و اجرا
- [[entities/orchestrator]] — ارزیابی ریسک با AI
- [[entities/data-layer]] — ذخیره نتیجه

## منابع کد
- `MyProjects/Octopus/README.md:95` — sequence diagram معاملات
