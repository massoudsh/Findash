'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

interface AuditEvent {
  id: number;
  timestamp: string;
  actor: string;
  action: string;
  target_type: string | null;
  target_id: string | null;
  detail: Record<string, unknown> | null;
  ip_address: string | null;
}

export default function AuditLogPage() {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('/api/admin/audit-log?limit=100', { cache: 'no-store' });
      if (!response.ok) throw new Error(response.status === 401 || response.status === 403
        ? 'برای مشاهده لاگ ممیزی باید با حساب مدیر وارد شوید.' : 'دریافت رویدادها ناموفق بود.');
      setEvents(await response.json());
    } catch (e) {
      setEvents([]);
      setError(e instanceof Error ? e.message : 'ارتباط با سرور برقرار نشد.');
    } finally { setLoading(false); }
  }, []);
  useEffect(() => { void load(); }, [load]);
  const filtered = events.filter(event => `${event.actor} ${event.action} ${event.target_type ?? ''} ${event.target_id ?? ''}`.toLowerCase().includes(search.toLowerCase()));
  return <section dir="rtl" className="space-y-4">
    <h1 className="text-2xl font-bold">لاگ ممیزی</h1>
    <p className="text-muted-foreground">حداکثر ۱۰۰ رویداد اخیر ثبت‌شده در پایگاه داده</p>
    <label className="block">جست‌وجوی رویداد<Input value={search} onChange={e => setSearch(e.target.value)} /></label>
    <Button onClick={() => void load()} disabled={loading}>بازخوانی</Button>
    {error && <div role="alert"><p className="text-destructive">{error}</p><Link className="underline" href="/auth/signin?callbackUrl=/admin">ورود مدیر</Link></div>}
    {loading ? <p role="status">در حال بارگذاری…</p> : !error && filtered.length === 0 ? <p>رویدادی پیدا نشد.</p> : filtered.map(event =>
      <article key={event.id} className="space-y-2 rounded-lg border p-4">
        <h2 className="font-semibold"><bdi>{event.action}</bdi></h2>
        <p><bdi>{event.actor}</bdi> · {new Date(event.timestamp).toLocaleString('fa-IR')}</p>
        <p>منبع: <bdi>{event.target_type ?? '—'} / {event.target_id ?? '—'}</bdi></p>
        <p>نشانی شبکه: <bdi>{event.ip_address ?? '—'}</bdi></p>
        {event.detail && <details><summary className="cursor-pointer">جزئیات تغییر</summary><pre dir="ltr" className="overflow-auto text-sm">{JSON.stringify(event.detail, null, 2)}</pre></details>}
      </article>
    )}
  </section>;
}
