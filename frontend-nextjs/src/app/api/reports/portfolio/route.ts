import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth-options';
import { getBackendUrl } from '@/lib/backend-url';

export async function GET(request: Request) {
  const session = await getServerSession(authOptions);
  if (!session?.accessToken) {
    return NextResponse.json(
      { detail: 'برای دریافت گزارش PDF باید وارد حساب خود شوید.' },
      { status: 401 }
    );
  }

  const url = new URL(request.url);
  const portfolioId = url.searchParams.get('portfolio_id');
  const params = new URLSearchParams();
  if (portfolioId?.match(/^\d+$/)) params.set('portfolio_id', portfolioId);

  try {
    const response = await fetch(`${getBackendUrl()}/api/reports/portfolio.pdf?${params}`, {
      headers: { Authorization: `Bearer ${session.accessToken}` },
      cache: 'no-store',
    });

    if (!response.ok) {
      return NextResponse.json(
        await response.json().catch(() => ({ detail: 'تولید گزارش ناموفق بود' })),
        { status: response.status }
      );
    }

    return new Response(await response.arrayBuffer(), {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': 'attachment; filename="portfolio-report.pdf"',
        'Cache-Control': 'no-store',
      },
    });
  } catch {
    return NextResponse.json({ detail: 'سرور گزارش در دسترس نیست' }, { status: 503 });
  }
}
