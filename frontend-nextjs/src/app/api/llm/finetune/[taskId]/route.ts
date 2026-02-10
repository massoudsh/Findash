import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function GET(
  request: Request,
  context: { params: Promise<{ taskId: string }> }
) {
  const params = await context.params;
  const taskId = params.taskId;

  try {
    const backendResponse = await fetch(`${BACKEND_URL}/llm/finetune/${taskId}`);
    
    if (!backendResponse.ok) {
      const errorData = await backendResponse.json();
      return NextResponse.json({ error: errorData.detail || 'Failed to fetch job status' }, { status: backendResponse.status });
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error(`Error proxying status check for task ${taskId}:`, error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
} 