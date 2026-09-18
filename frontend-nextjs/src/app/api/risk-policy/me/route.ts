import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import { getBackendUrl } from '@/lib/backend-url';

const unauthorized = () => NextResponse.json({ detail: 'برای این بخش باید وارد حساب خود شوید.' }, { status: 401 });

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session) return unauthorized();
  try {
    const resp = await fetch(`${getBackendUrl()}/api/risk-policy/me`, {
      headers: { Authorization: `Bearer ${session.accessToken}` }, cache: 'no-store',
    });
    return NextResponse.json(await resp.json().catch(() => ({ detail: 'خطای سرور' })), { status: resp.status });
  } catch {
    return NextResponse.json({ detail: 'سرور در دسترس نیست' }, { status: 503 });
  }
}

export async function PUT(request: Request) {
  const session = await getServerSession(authOptions);
  if (!session) return unauthorized();
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: 'بدنه درخواست نامعتبر است' }, { status: 400 });
  }
  try {
    const resp = await fetch(`${getBackendUrl()}/api/risk-policy/me`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${session.accessToken}` },
      body: JSON.stringify(body),
    });
    return NextResponse.json(await resp.json().catch(() => ({ detail: 'خطای سرور' })), { status: resp.status });
  } catch {
    return NextResponse.json({ detail: 'سرور در دسترس نیست' }, { status: 503 });
  }
}
