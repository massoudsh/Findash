import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { getBackendUrl } from '@/lib/backend-url';

// Proxy to backend /api/admin/audit-log (issue #12) — لاگ ممیزی واقعی به‌جای داده mock.
export async function GET(request: Request) {
  const session = await getServerSession();
  if (!session) {
    return NextResponse.json({ detail: 'احراز هویت لازم است' }, { status: 401 });
  }
  if (session.user?.role !== 'admin') {
    return NextResponse.json({ detail: 'دسترسی محدود به مدیران سیستم است' }, { status: 403 });
  }

  const backendUrl = getBackendUrl();
  const limit = new URL(request.url).searchParams.get('limit') ?? '100';
  try {
    const token = (session as { accessToken?: string }).accessToken;
    const resp = await fetch(`${backendUrl}/api/admin/audit-log?limit=${encodeURIComponent(limit)}`, {
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
