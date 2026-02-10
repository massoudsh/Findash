import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { output_dir } = body;

    if (!output_dir) {
      return NextResponse.json({ error: 'output_dir is required' }, { status: 400 });
    }

    const backendResponse = await fetch(`${BACKEND_URL}/llm/finetune`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ output_dir }),
    });

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json();
      return NextResponse.json({ error: errorData.detail || 'Failed to start fine-tuning job' }, { status: backendResponse.status });
    }

    const data = await backendResponse.json();
    return NextResponse.json(data, { status: 202 });

  } catch (error) {
    console.error('Error proxying to backend:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
} 