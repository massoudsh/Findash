import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  try {
    const { symbol } = await params;
    if (!symbol) {
      return NextResponse.json({ error: 'Symbol required' }, { status: 400 });
    }
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get('limit') ?? '100';
    const res = await fetch(
      `${BACKEND_URL}/api/funding-rate/${encodeURIComponent(symbol)}/history?limit=${limit}`,
      {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(15000),
      }
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return NextResponse.json(data || { error: res.statusText }, { status: res.status });
    }
    return NextResponse.json(data);
  } catch (err) {
    console.error('Funding history error:', err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Failed to fetch funding history' },
      { status: 502 }
    );
  }
}
