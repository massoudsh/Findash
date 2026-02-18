import { NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/llm/status`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
    })
    if (!res.ok) {
      return NextResponse.json(
        { any_configured: false, falcon_configured: false, fingpt_local_configured: false, hf_configured: false },
        { status: 200 }
      )
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch {
    return NextResponse.json(
      { any_configured: false, falcon_configured: false, fingpt_local_configured: false, hf_configured: false },
      { status: 200 }
    )
  }
}
