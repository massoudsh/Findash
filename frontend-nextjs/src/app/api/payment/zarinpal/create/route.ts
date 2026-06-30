import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { getBackendUrl } from '@/lib/backend-url';

export async function POST(request: NextRequest) {
  const session = await getServerSession();
  if (!session) {
    return NextResponse.json({ detail: 'احراز هویت لازم است' }, { status: 401 });
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: 'بدنه درخواست نامعتبر است' }, { status: 400 });
  }

  const backendUrl = getBackendUrl();
  try {
    const token = (session as { accessToken?: string }).accessToken;
    const resp = await fetch(`${backendUrl}/api/payment/zarinpal/create`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
    });

    const data = await resp.json().catch(() => ({ detail: 'خطای سرور' }));
    return NextResponse.json(data, { status: resp.status });
  } catch {
    return NextResponse.json({ detail: 'سرور در دسترس نیست' }, { status: 503 });
  }
}
