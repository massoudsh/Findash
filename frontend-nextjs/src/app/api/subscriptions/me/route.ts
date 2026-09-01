import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { getBackendUrl } from '@/lib/backend-url';

export async function GET() {
  const session = await getServerSession();
  if (!session) {
    return NextResponse.json({ detail: 'احراز هویت لازم است' }, { status: 401 });
  }

  const backendUrl = getBackendUrl();
  try {
    const token = (session as { accessToken?: string }).accessToken;
    const resp = await fetch(`${backendUrl}/api/subscriptions/me`, {
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
