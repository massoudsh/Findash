import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import { getBackendUrl } from '@/lib/backend-url';

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session) {
    return NextResponse.json({ detail: 'برای این بخش باید وارد حساب خود شوید.' }, { status: 401 });
  }
  try {
    const resp = await fetch(`${getBackendUrl()}/api/risk-policy/breaches`, {
      headers: { Authorization: `Bearer ${session.accessToken}` }, cache: 'no-store',
    });
    return NextResponse.json(await resp.json().catch(() => ({ detail: 'خطای سرور' })), { status: resp.status });
  } catch {
    return NextResponse.json({ detail: 'سرور در دسترس نیست' }, { status: 503 });
  }
}
