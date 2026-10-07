# Data ↔ Reports Integration

> صفحه `/data` (نمودارها) و صفحه `/reports` (گزارش AI) به‌صورت additive به هم وصل شدند تا داده‌ی بصری و تفسیر هوش مصنوعی همان داده در یک تجربه‌ی واحد در دسترس باشد — بدون حذف یا جابه‌جایی هیچ صفحه‌ی مستقل قبلی.

## چرا
داده (نمودار/چارت رنگی) و گزارش (تفسیر متنی/AI-generated) دو روی یک سکه‌اند: کاربر اول ترکیب/روند را بصری می‌بیند، بعد می‌خواهد تفسیر آماده (حتی کاملاً AI-generated) از همان داده بگیرد.

## تغییرات
| فایل | تغییر |
|------|-------|
| `frontend-nextjs/src/app/data/page.tsx` | تب سوم `AI Report` اضافه شد (کنار `Explorer`/`Charts`) — `ReportsContent` موجود (`components/reports/reports-content.tsx`) را reuse می‌کند، بدون duplicate logic. state تب در query param `?tab=explorer\|charts\|report` سینک است. دکمه‌ی «Turn this data into an AI-written report» در پایین تب Charts کاربر را مستقیم به تب Report می‌برد. |
| `frontend-nextjs/src/app/reports/page.tsx` | لینک برگشتی «View the underlying charts this report is based on» → `/data?tab=charts` اضافه شد (ناوبری دوطرفه). |

## معماری (بدون تغییر backend)
گزارش AI همان مسیر قبلی را طی می‌کند (`ReportsContent` → `POST /api/llm/reports/generate-insights` → LLM واقعی Falcon/FinGPT یا simulated fallback، جزئیات در `LlmStatusBadge`/`/api/llm/status`). این تغییر فقط لایه‌ی presentation/routing فرانت‌اند است.

## جایگاه در UI
- `/data` → تب `AI Report` (query: `?tab=report`)
- `/reports` → صفحه مستقل قبلی، بدون تغییر (additive، حذف نشد)

## اجزای درگیر
- [[entities/frontend]] — صفحات `/data` و `/reports`، کامپوننت `ReportsContent`/`LlmStatusBadge`

## منابع کد
- `frontend-nextjs/src/app/data/page.tsx`
- `frontend-nextjs/src/app/reports/page.tsx`
- `frontend-nextjs/src/components/reports/reports-content.tsx`
- `frontend-nextjs/src/components/reports/llm-status-badge.tsx`
