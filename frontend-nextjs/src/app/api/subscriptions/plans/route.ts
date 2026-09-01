import { NextResponse } from 'next/server';
import { getBackendUrl } from '@/lib/backend-url';

export async function GET() {
  const backendUrl = getBackendUrl();
  try {
    const resp = await fetch(`${backendUrl}/api/subscriptions/plans`, {
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
    });
    const data = await resp.json().catch(() => ({ detail: 'خطای سرور' }));
    return NextResponse.json(data, { status: resp.status });
  } catch {
    return NextResponse.json({ detail: 'سرور در دسترس نیست' }, { status: 503 });
  }
}
