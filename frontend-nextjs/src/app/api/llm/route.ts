import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { action, text, output_dir } = body

    let endpoint = ''
    let payload = {}

    switch (action) {
      case 'finetune':
        endpoint = '/llm/finetune'
        payload = { output_dir: output_dir || './peft-output' }
        break
      case 'predict':
        endpoint = '/llm/predict'
        payload = { text }
        break
      case 'llama-predict':
        endpoint = '/llm/llama/predict'
        payload = { text }
        break
      case 'generate-insights':
        endpoint = '/llm/reports/generate-insights'
        payload = {}
        break
      default:
        return NextResponse.json(
          { error: 'Invalid action' },
          { status: 400 }
        )
    }

    const response = await fetch(`${BACKEND_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    })

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('LLM API Error:', error)
    return NextResponse.json(
      { error: 'Failed to process LLM request' },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const action = searchParams.get('action')
    const taskId = searchParams.get('taskId')

    let endpoint = ''

    switch (action) {
      case 'status':
        if (taskId) {
          endpoint = `/llm/finetune/${taskId}`
        } else {
          endpoint = '/llm/finetune'
        }
        break
      case 'monitoring':
        endpoint = '/llm/monitoring'
        break
      case 'analysis-status':
        endpoint = '/llm/reports/analysis-status'
        break
      case 'data-sources':
        endpoint = '/llm/reports/data-sources'
        break
      default:
        return NextResponse.json(
          { error: 'Invalid action' },
          { status: 400 }
        )
    }

    const response = await fetch(`${BACKEND_URL}${endpoint}`)
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('LLM Status API Error:', error)
    return NextResponse.json(
      { error: 'Failed to get LLM status' },
      { status: 500 }
    )
  }
} 