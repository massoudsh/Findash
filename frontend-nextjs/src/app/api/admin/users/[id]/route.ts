import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import { getBackendUrl } from '@/lib/backend-url';

// Proxy to backend PATCH /api/admin/users/{id} (issue #12) — تغییر نقش/فعال‌بودن.
export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await getServerSession(authOptions);
  if (!session) {
    return NextResponse.json({ detail: 'احراز هویت لازم است' }, { status: 401 });
  }
  if (session.user?.role !== 'admin') {
    return NextResponse.json({ detail: 'دسترسی محدود به مدیران سیستم است' }, { status: 403 });
  }

  const { id } = await params;
  if (!/^\d+$/.test(id)) {
    return NextResponse.json({ detail: 'شناسه نامعتبر است' }, { status: 400 });
  }
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: 'بدنه درخواست نامعتبر است' }, { status: 400 });
  }
  const backendUrl = getBackendUrl();
  try {
    const token = session.accessToken;
    const resp = await fetch(`${backendUrl}/api/admin/users/${id}`, {
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
