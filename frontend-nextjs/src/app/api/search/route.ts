import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

/** Platform index for fast search when backend Elasticsearch is not available */
const PLATFORM_INDEX: { title: string; path: string; keywords: string[]; type: string }[] = [
  { title: 'Dashboard', path: '/dashboard', keywords: ['home', 'overview', 'analytics'], type: 'page' },
  { title: 'مرکز فرماندهی', path: '/trading', keywords: ['trading', 'market', 'bots', 'options', 'command center', 'مرکز فرماندهی'], type: 'page' },
  { title: 'Options', path: '/options', keywords: ['options', 'greeks', 'derivatives'], type: 'page' },
  { title: 'Live Trading', path: '/trades', keywords: ['trades', 'orders', 'positions', 'live'], type: 'page' },
  { title: 'Portfolio', path: '/portfolio', keywords: ['portfolio', 'holdings', 'allocation'], type: 'page' },
  { title: 'Strategies', path: '/strategies', keywords: ['strategies', 'backtest', 'signals'], type: 'page' },
  { title: 'Risk Assessment', path: '/trading?tab=risk', keywords: ['risk', 'var', 'compliance'], type: 'page' },
  { title: 'Backtesting', path: '/strategies?tab=backtesting', keywords: ['backtest', 'history', 'strategy'], type: 'page' },
  { title: 'Paper Trading', path: '/paper-trading', keywords: ['paper', 'simulate', 'practice'], type: 'page' },
  { title: 'Market Data', path: '/market-data', keywords: ['market', 'quotes', 'stocks', 'crypto'], type: 'page' },
  { title: 'Real-time', path: '/realtime', keywords: ['realtime', 'stream', 'live', 'alerts'], type: 'page' },
  { title: 'Technical', path: '/technical', keywords: ['technical', 'charts', 'indicators'], type: 'page' },
  { title: 'Fundamental Research', path: '/fundamental-data', keywords: ['fundamental', 'earnings', 'valuation'], type: 'page' },
  { title: 'Macro', path: '/macro', keywords: ['macro', 'rates', 'economy'], type: 'page' },
  { title: 'On-chain', path: '/on-chain', keywords: ['onchain', 'blockchain', 'crypto'], type: 'page' },
  { title: 'Social Signals', path: '/social', keywords: ['social', 'sentiment', 'twitter'], type: 'page' },
  { title: 'AI Models', path: '/ai-models', keywords: ['ai', 'ml', 'models', 'training'], type: 'page' },
  { title: 'Data Explorer', path: '/data-explorer', keywords: ['data', 'export', 'explorer'], type: 'page' },
  { title: 'Visualization', path: '/visualization', keywords: ['charts', 'visualization', 'analytics'], type: 'page' },
  { title: 'Reports', path: '/reports', keywords: ['reports', 'insights', 'llm'], type: 'page' },
  { title: 'Notifications', path: '/notifications', keywords: ['notifications', 'alerts'], type: 'page' },
  { title: 'Settings', path: '/settings', keywords: ['settings', 'preferences'], type: 'page' },
  { title: 'Profile', path: '/profile', keywords: ['profile', 'account'], type: 'page' },
  { title: 'Help', path: '/help', keywords: ['help', 'docs', 'support'], type: 'page' },
];

function localSearch(q: string): { id: string; title: string; path: string; type: string }[] {
  if (!q || q.length < 2) return [];
  const lower = q.toLowerCase().trim();
  return PLATFORM_INDEX.filter(
    (item) =>
      item.title.toLowerCase().includes(lower) ||
      item.path.toLowerCase().includes(lower) ||
      item.keywords.some((k) => k.includes(lower))
  )
    .slice(0, 12)
    .map((item) => ({
      id: `platform-${item.path}`,
      title: item.title,
      path: item.path,
      type: item.type,
    }));
}

/**
 * Platform search API. Uses backend Elasticsearch when available (BACKEND_URL/api/search?q=),
 * otherwise returns fast results from a local platform index.
 */
export async function GET(request: NextRequest) {
  const q = request.nextUrl.searchParams.get('q')?.trim() || '';
  if (q.length < 2) {
    return NextResponse.json({ results: [] });
  }

  try {
    const res = await fetch(
      `${BACKEND_URL}/api/search?q=${encodeURIComponent(q)}&limit=15`,
      { signal: AbortSignal.timeout(2000), cache: 'no-store' }
    );
    if (res.ok) {
      const data = await res.json();
      return NextResponse.json(data);
    }
  } catch {
    // Backend unavailable or no Elasticsearch; use local index
  }

  const results = localSearch(q);
  return NextResponse.json({ results });
}
