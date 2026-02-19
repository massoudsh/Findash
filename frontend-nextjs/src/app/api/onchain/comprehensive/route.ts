import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || process.env.BACKEND_URL || 'http://localhost:8000';

/**
 * Proxy to backend real on-chain data (Bitcoin network, flows, valuation, DeFi).
 * When backend is up, data is from chain/indexer sources; otherwise frontend can use mock.
 */
export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/onchain/onchain/comprehensive`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(20000),
    });
    if (!res.ok) {
      return NextResponse.json(
        { error: 'On-chain service unavailable', status: res.status },
        { status: 502 }
      );
    }
    const json = await res.json();
    const data = json.data ?? json;
    const last_updated = data?.last_updated ?? new Date().toISOString();
    return NextResponse.json({ data, last_updated });
  } catch (error) {
    console.error('Onchain comprehensive proxy error:', error);
    return NextResponse.json(
      { error: 'On-chain service unavailable' },
      { status: 502 }
    );
  }
}
