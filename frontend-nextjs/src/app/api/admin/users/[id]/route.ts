import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { getBackendUrl } from '@/lib/backend-url';

// Proxy to backend PATCH /api/admin/users/{id} (issue #12) — تغییر نقش/فعال‌بودن.
export async function PATCH(
  request: Request,
  { params }: { params: { id: string } }
) {
  const session = await getServerSession();
  if (!session) {
    return NextResponse.json({ detail: 'احراز هویت لازم است' }, { status: 401 });
  }
  if (session.user?.role !== 'admin') {
    return NextResponse.json({ detail: 'دسترسی محدود به مدیران سیستم است' }, { status: 403 });
  }

  const backendUrl = getBackendUrl();
  try {
    const token = (session as { accessToken?: string }).accessToken;
    const body = await request.json().catch(() => ({}));
    const resp = await fetch(`${backendUrl}/api/admin/users/${params.id}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
      cache: 'no-store',
    });
    const data = await resp.json().catch(() => ({ detail: 'خطای سرور' }));
    return NextResponse.json(data, { status: resp.status });
  } catch {
    return NextResponse.json({ detail: 'سرور در دسترس نیست' }, { status: 503 });
  }
}
