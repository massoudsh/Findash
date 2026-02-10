/**
 * Quantum Trading Matrix™ API Proxy
 * Next.js API routes that proxy requests to FastAPI backend
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// Generic proxy function
async function proxyRequest(request: NextRequest, endpoint: string) {
  try {
    const url = new URL(request.url);
    const searchParams = url.searchParams.toString();
    const backendUrl = `${BACKEND_URL}${endpoint}${searchParams ? `?${searchParams}` : ''}`;

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    // Forward authorization header if present
    const authHeader = request.headers.get('authorization');
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }

    const requestOptions: RequestInit = {
      method: request.method,
      headers,
    };

    // Add body for POST/PUT requests
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      const body = await request.text();
      if (body) {
        requestOptions.body = body;
      }
    }

    const response = await fetch(backendUrl, requestOptions);
    const data = await response.json();

    return NextResponse.json(data, { 
      status: response.status,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      }
    });

  } catch (error) {
    console.error('API Proxy Error:', error);
    return NextResponse.json(
      { error: 'Internal server error', message: String(error) },
      { status: 500 }
    );
  }
}

// Market Data Routes
export async function GET(request: NextRequest) {
  const url = new URL(request.url);
  const path = url.pathname.replace('/api/quantum-trading', '');
  
  if (path.startsWith('/market-data/')) {
    return proxyRequest(request, path);
  }
  
  if (path.startsWith('/options/')) {
    return proxyRequest(request, path);
  }
  
  if (path.startsWith('/portfolios/')) {
    return proxyRequest(request, path);
  }
  
  if (path.startsWith('/realtime/')) {
    return proxyRequest(request, path);
  }
  
  if (path === '/health') {
    return proxyRequest(request, '/health');
  }
  
  if (path === '/system/metrics') {
    return proxyRequest(request, '/system/metrics');
  }
  
  if (path === '/trading-pods/status') {
    return proxyRequest(request, '/trading-pods/status');
  }

  return NextResponse.json({ error: 'Not found' }, { status: 404 });
}

// Analysis and Trading Routes
export async function POST(request: NextRequest) {
  const url = new URL(request.url);
  const path = url.pathname.replace('/api/quantum-trading', '');
  
  if (path === '/options/analyze') {
    return proxyRequest(request, '/options/analyze');
  }
  
  if (path === '/alternative-data/analyze') {
    return proxyRequest(request, '/alternative-data/analyze');
  }
  
  if (path === '/compliance/check') {
    return proxyRequest(request, '/compliance/check');
  }
  
  if (path === '/esg/analyze') {
    return proxyRequest(request, '/esg/analyze');
  }
  
  if (path === '/quantum/predict') {
    return proxyRequest(request, '/quantum/predict');
  }

  return NextResponse.json({ error: 'Not found' }, { status: 404 });
}

// Handle OPTIONS for CORS
export async function OPTIONS(request: NextRequest) {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
} 