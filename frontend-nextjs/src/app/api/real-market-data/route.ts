import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

function chunkArray<T>(arr: T[], size: number): T[][] {
  const res: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    res.push(arr.slice(i, i + size));
  }
  return res;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbols = searchParams.get('symbols') || 'AAPL,TSLA,MSFT,GOOGL,NVDA,BTC-USD,ETH-USD';
    const symbolList = symbols.split(',').map(s => s.trim()).filter(Boolean);
    const batches = chunkArray(symbolList, 10);
    let allData: Record<string, any> = {};
    let anySuccess = false;
    for (const batch of batches) {
      const batchSymbols = batch.join(',');
      try {
        const backendResponse = await fetch(
          `${BACKEND_URL}/api/simple-market-data/real-time?symbols=${batchSymbols}`,
          {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
            signal: AbortSignal.timeout(10000),
          }
        );
        if (backendResponse.ok) {
          const data = await backendResponse.json();
          if (data.data) {
            Object.assign(allData, data.data);
            anySuccess = true;
          }
        } else {
          console.error(`Backend responded with status: ${backendResponse.status} for symbols: ${batchSymbols}`);
        }
      } catch (err) {
        console.error(`Error fetching batch: ${batchSymbols}`, err);
      }
    }
    if (anySuccess) {
      return NextResponse.json({
        status: 'success',
        data: allData,
        timestamp: new Date().toISOString(),
        symbols_requested: symbolList.length,
        symbols_returned: Object.keys(allData).length,
      });
    }
    // Fallback mock data if all batches fail
    const mockData = {
      status: 'error_fallback',
      data: {
        'AAPL': {
          symbol: 'AAPL', price: 175.23, open: 173.45, high: 176.89, low: 172.11, volume: 65432100, change: 1.78, change_percent: 1.03, timestamp: new Date().toISOString(), source: 'mock_fallback'
        },
        'BTC-USD': {
          symbol: 'BTC-USD', price: 107850.00, open: 108100.00, high: 108300.00, low: 107600.00, volume: 3400, change: -250.00, change_percent: -0.23, timestamp: new Date().toISOString(), source: 'mock_fallback'
        }
      },
      timestamp: new Date().toISOString(),
      symbols_requested: symbolList.length,
      symbols_returned: 2,
      note: 'Fallback mock data due to backend error'
    };
    return NextResponse.json(mockData);
  } catch (error) {
    console.error('Error fetching real market data:', error);
    // Fallback mock data
    const mockData = {
      status: 'error_fallback',
      data: {
        'AAPL': {
          symbol: 'AAPL', price: 175.23, open: 173.45, high: 176.89, low: 172.11, volume: 65432100, change: 1.78, change_percent: 1.03, timestamp: new Date().toISOString(), source: 'mock_fallback'
        },
        'BTC-USD': {
          symbol: 'BTC-USD', price: 107850.00, open: 108100.00, high: 108300.00, low: 107600.00, volume: 3400, change: -250.00, change_percent: -0.23, timestamp: new Date().toISOString(), source: 'mock_fallback'
        }
      },
      timestamp: new Date().toISOString(),
      symbols_requested: 2,
      symbols_returned: 2,
      error: 'Backend connection failed, using fallback data'
    };
    return NextResponse.json(mockData);
  }
} 