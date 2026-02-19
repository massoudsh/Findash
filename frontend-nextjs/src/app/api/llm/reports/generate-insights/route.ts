import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function POST(request: Request) {
  try {
    const body = await request.json().catch(() => ({}));
    const backendResponse = await fetch(`${BACKEND_URL}/llm/reports/generate-insights`, {
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
      ? 'Backend unreachable. Start the API (e.g. run Docker or uvicorn) and set BACKEND_URL if needed.'
      : error instanceof Error
        ? error.message
        : 'Internal Server Error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
} 