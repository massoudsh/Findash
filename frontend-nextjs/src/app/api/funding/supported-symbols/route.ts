import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/funding-rate/supported-symbols`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(5000),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      return NextResponse.json(data || { error: res.statusText }, { status: res.status });
    }
    return NextResponse.json(data);
  } catch (err) {
    console.error('Supported symbols error:', err);
    return NextResponse.json(
      {
        supported_symbols: ['BTCUSDT', 'ETHUSDT'],
        total_count: 2,
        exchange: 'binance_futures',
        note: 'Fallback list; backend unreachable',
      },
      { status: 200 }
    );
  }
}
