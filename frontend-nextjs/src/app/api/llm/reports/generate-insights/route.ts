import { NextResponse } from 'next/server';

import { getBackendUrl } from '@/lib/backend-url';

export async function POST(request: Request) {
  try {
    const body = await request.json().catch(() => ({}));
    const backendResponse = await fetch(`${getBackendUrl()}/llm/reports/generate-insights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({}));
      return NextResponse.json(
        { error: (errorData as { detail?: string }).detail || 'Failed to generate AI insights' },
        { status: backendResponse.status }
      );
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('Error proxying AI insights request:', error);
    const err = error as NodeJS.ErrnoException & { cause?: { code?: string } };
    const isConnectionRefused =
      err?.code === 'ECONNREFUSED' || err?.cause?.code === 'ECONNREFUSED';
    const message = isConnectionRefused
      ? 'Backend unreachable. Run ./scripts/start-dev.sh or start the API (e.g. python start.py) and set NEXT_PUBLIC_API_URL in .env.local if needed.'
      : error instanceof Error
        ? error.message
        : 'Internal Server Error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
} 