import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function GET(
  request: Request,
  context: { params: Promise<{ portfolioId: string }> }
) {
  const params = await context.params;
  const portfolioId = params.portfolioId;

  try {
    const backendResponse = await fetch(`${BACKEND_URL}/portfolio/${portfolioId}/positions`);
    
    if (!backendResponse.ok) {
      const errorData = await backendResponse.json();
      return NextResponse.json({ error: errorData.detail || 'Failed to fetch positions' }, { status: backendResponse.status });
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error(`Error proxying positions request for portfolio ${portfolioId}:`, error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
} 