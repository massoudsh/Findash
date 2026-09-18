import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import { getBackendUrl } from '@/lib/backend-url';

// Proxy to backend /api/admin/users (issue #12). دسترسی فقط برای role=admin؛
// ادمین‌بودن هم از session چک می‌شود تا مهمان حتی یک درخواست به بک‌اند نفرستد.
export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session) {
    return NextResponse.json({ detail: 'احراز هویت لازم است' }, { status: 401 });
  }
  if (session.user?.role !== 'admin') {
    return NextResponse.json({ detail: 'دسترسی محدود به مدیران سیستم است' }, { status: 403 });
  }

  const backendUrl = getBackendUrl();
  try {
    const token = (session as { accessToken?: string }).accessToken;
    const resp = await fetch(`${backendUrl}/api/admin/users`, {
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      cache: 'no-store',
    });
    const data = await resp.json().catch(() => ({ detail: 'خطای سرور' }));
    return NextResponse.json(data, { status: resp.status });
  } catch {
    return NextResponse.json({ detail: 'سرور در دسترس نیست' }, { status: 503 });
  }
}
