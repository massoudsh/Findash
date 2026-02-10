import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  try {
    const { symbol } = await params;
    if (!symbol) {
      return NextResponse.json({ error: 'Symbol required' }, { status: 400 });
    }
    const res = await fetch(`${BACKEND_URL}/api/funding-rate/${encodeURIComponent(symbol)}/refresh`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(15000),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return NextResponse.json(data || { error: res.statusText }, { status: res.status });
    }
    return NextResponse.json(data);
  } catch (err) {
    console.error('Funding refresh error:', err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Failed to fetch funding rate' },
      { status: 502 }
    );
  }
}
