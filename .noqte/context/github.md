
# GitHub

ریپو: `massoudsh/Findash` — برنچ فعلی: `main`

اعتبار github داخل کانتینر از قبل ست شده — `git pull`, `git push`, `git fetch` بدون pass-prompt کار می‌کند.
دستورات کلیدی (از `/project/`):
- `git status` — قبل از هر commit وضعیت را ببین
- `git add -A && git commit -m "<پیام معنادار>"` — کامیت تغییرات
- `git push origin main` — push روی برنچ فعلی
- `git pull --rebase origin main` — sync با remote قبل از push
- `git checkout -b <نام-برنچ>` — برنچ جدید برای تغییرات experimental

**رفتار خواسته‌شده:**
- اگر کاربر صریح گفت «push کن» یا «commit کن»، فوراً انجام بده — نیاز به تأیید مجدد نیست.
- پس از یک تغییر بزرگ (چند فایل، feature کامل)، خودت پیشنهاد commit بده و در صورت تأیید کاربر push کن.
- پیام کامیت **معنادار** بنویس (Conventional Commits بهتر): `feat: ...`, `fix: ...`, `refactor: ...`. هرگز «update»، «wip» یا «.» ننویس.
- اگر `git pull` با conflict مواجه شد، اطلاع بده و راهنمایی بخواه — خودت random conflict-resolve نکن.
- اگر کاربر گفت روی برنچ دیگری کار کن، اول `git fetch` بزن و سپس `git checkout`. اگر برنچ remote موجود نبود، بپرس آیا برنچ جدید بسازی.
