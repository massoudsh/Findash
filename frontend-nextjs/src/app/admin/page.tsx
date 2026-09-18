'use client';

import { useCallback, useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { StartupTrackerPanel } from '@/components/admin/startup-tracker-panel';
import AuditLogPage from '@/app/audit-log/page';

interface AdminUser {
  id: number;
  name: string;
  email: string;
  role: string;
  is_active: boolean;
  last_login: string | null;
  total_trades: number;
}

const roles: Record<string, string> = { admin: 'مدیر', trader: 'معامله‌گر', demo: 'دمو' };

export default function AdminPage() {
  const { data: session } = useSession();
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [pending, setPending] = useState<number | null>(null);
  const [search, setSearch] = useState('');
  const [role, setRole] = useState('all');
  const [status, setStatus] = useState('all');
  const [revision, setRevision] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('/api/admin/users', { cache: 'no-store' });
      if (!response.ok) throw new Error(response.status === 403 ? 'دسترسی محدود به مدیران است.' : 'دریافت کاربران ناموفق بود. دوباره تلاش کنید.');
      setUsers(await response.json());
    } catch (e) {
      setUsers([]);
      setError(e instanceof Error ? e.message : 'ارتباط با سرور برقرار نشد.');
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => { void load(); }, [load]);

  async function update(user: AdminUser, patch: { role?: string; is_active?: boolean }) {
    if (!window.confirm(`تغییر دسترسی ${user.name} را تأیید می‌کنید؟`)) return;
    setPending(user.id);
    setError('');
    try {
      const response = await fetch(`/api/admin/users/${user.id}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'تغییر کاربر ناموفق بود.');
      setUsers(previous => previous.map(row => row.id === user.id ? result : row));
      setRevision(value => value + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'ارتباط با سرور برقرار نشد.');
    } finally {
      setPending(null);
    }
  }

  const filtered = users.filter(user =>
    `${user.name} ${user.email}`.toLowerCase().includes(search.toLowerCase()) &&
    (role === 'all' || user.role === role) &&
    (status === 'all' || user.is_active === (status === 'active'))
  );

  return <main dir="rtl" className="space-y-6">
    <h1 className="text-3xl font-bold">پنل مدیریت</h1>
    <p className="text-muted-foreground">کاربران و رویدادهای واقعی سامانه؛ دسترسی فقط برای مدیران.</p>
    {error && <p role="alert" className="text-destructive">{error}</p>}
    <Tabs defaultValue="users">
      <TabsList className="flex h-auto flex-wrap">
        <TabsTrigger value="users">کاربران</TabsTrigger>
        <TabsTrigger value="audit">لاگ ممیزی</TabsTrigger>
        <TabsTrigger value="startup">استارتاپ‌تراکر</TabsTrigger>
      </TabsList>
      <TabsContent value="users">
        <Card>
          <CardHeader><CardTitle>مدیریت کاربران</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-end gap-3">
              <label className="flex-1 min-w-48">جست‌وجو<Input value={search} onChange={e => setSearch(e.target.value)} /></label>
              <label>نقش<select className="block rounded border bg-background p-2" value={role} onChange={e => setRole(e.target.value)}>
                <option value="all">همه نقش‌ها</option>{Object.entries(roles).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
              </select></label>
              <label>وضعیت<select className="block rounded border bg-background p-2" value={status} onChange={e => setStatus(e.target.value)}>
                <option value="all">همه</option><option value="active">فعال</option><option value="inactive">غیرفعال</option>
              </select></label>
              <Button disabled={loading || pending !== null} onClick={() => void load()}>بازخوانی</Button>
            </div>
            {loading ? <p role="status">در حال بارگذاری…</p> : <>
              <p>{users.length.toLocaleString('fa-IR')} کاربر؛ {users.filter(u => u.is_active).length.toLocaleString('fa-IR')} فعال</p>
              {!error && filtered.length === 0 && <p>کاربری با این فیلترها پیدا نشد.</p>}
              {filtered.map(user => <section key={user.id} className="space-y-3 rounded-lg border p-4">
                <h2 className="font-semibold">{user.name}</h2><p><bdi>{user.email}</bdi></p>
                <p>{user.is_active ? 'فعال' : 'غیرفعال'} · معاملات: {user.total_trades.toLocaleString('fa-IR')}</p>
                <p>آخرین ورود: {user.last_login ? new Date(user.last_login).toLocaleString('fa-IR') : 'ثبت نشده'}</p>
                <div className="flex flex-wrap gap-3">
                  <label>نقش کاربر<select aria-label={`نقش ${user.name}`} className="ms-2 rounded border bg-background p-2" value={user.role}
                    disabled={pending !== null || String(user.id) === session?.user.id}
                    onChange={e => void update(user, { role: e.target.value })}>
                    {Object.entries(roles).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
                  </select></label>
                  <Button variant="outline" disabled={pending !== null || String(user.id) === session?.user.id}
                    onClick={() => void update(user, { is_active: !user.is_active })}>
                    {pending === user.id ? 'در حال ذخیره…' : user.is_active ? 'غیرفعال کردن' : 'فعال کردن'}
                  </Button>
                </div>
              </section>)}
            </>}
          </CardContent>
        </Card>
      </TabsContent>
      <TabsContent value="audit"><AuditLogPage key={revision} /></TabsContent>
      <TabsContent value="startup"><StartupTrackerPanel /></TabsContent>
    </Tabs>
  </main>;
}
