import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { symbol, model_types, epochs } = body

    // Forward request to backend
    const response = await fetch(`${BACKEND_URL}/models/train-deep-learning`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbol,
        model_types: model_types || ['transformer', 'tcn'],
        epochs: epochs || 50
      })
    })

    const data = await response.json()
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('Training API Error:', error)
    return NextResponse.json(
      { error: 'Failed to start training' },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const taskId = searchParams.get('taskId')

    if (taskId) {
      // Get specific task status
      const response = await fetch(`${BACKEND_URL}/tasks/${taskId}`)
      const data = await response.json()
      return NextResponse.json(data)
    } else {
      // Get all training models
      const response = await fetch(`${BACKEND_URL}/models`)
      const data = await response.json()
      return NextResponse.json(data)
    }
  } catch (error) {
    console.error('Training Status API Error:', error)
    return NextResponse.json(
      { error: 'Failed to get training status' },
      { status: 500 }
    )
  }
} 