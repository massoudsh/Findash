import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || process.env.BACKEND_URL || 'http://localhost:8000';

/**
 * Proxy to backend real social sentiment (Fear & Greed, Reddit, Twitter, news).
 * When backend is up, data is real-time from sources; otherwise frontend falls back to mock.
 */
export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/social/social/comprehensive`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) {
      return NextResponse.json(
        { error: 'Social service unavailable', status: res.status },
        { status: 502 }
      );
    }
    const json = await res.json();
    const data = json.data ?? json;
    const last_updated = data?.last_updated ?? new Date().toISOString();
    return NextResponse.json({ data, last_updated });
  } catch (error) {
    console.error('Social comprehensive proxy error:', error);
    return NextResponse.json(
      { error: 'Social service unavailable' },
      { status: 502 }
    );
  }
}
