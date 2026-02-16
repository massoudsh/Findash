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
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
} 