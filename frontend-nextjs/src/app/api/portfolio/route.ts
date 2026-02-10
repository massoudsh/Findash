import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function GET() {
  try {
    const backendResponse = await fetch(`${BACKEND_URL}/portfolio`);
    
    if (!backendResponse.ok) {
      const errorData = await backendResponse.json();
      return NextResponse.json({ error: errorData.detail || 'Failed to fetch portfolios' }, { status: backendResponse.status });
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('Error proxying to backend:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
} 