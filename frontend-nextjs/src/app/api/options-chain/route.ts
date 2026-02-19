/**
 * Options chain API: real numbers for BTC/ETH (Deribit) and equities (backend yfinance).
 * GET /api/options-chain?symbol=BTC&expiry=25SEP25
 */

import { NextRequest, NextResponse } from 'next/server';
import { getBackendUrl } from '@/lib/backend-url';

const DERIBIT_BASE = 'https://www.deribit.com/api/v2/public';

interface DeribitBookSummary {
  instrument_name: string;
  bid_price: number | null;
  ask_price: number | null;
  mark_price: number;
  mark_iv?: number | null;
  underlying_price?: number;
  open_interest?: number;
  volume?: number;
}

/** Parse Deribit instrument name: BTC-25SEP25-50000-C → { expiryStr, strike, type } */
function parseDeribitName(name: string): { expiryStr: string; strike: number; type: 'call' | 'put' } | null {
  const match = name.match(/^(BTC|ETH)-(\d{2}[A-Z]{3}\d{2})-(\d+)-(C|P)$/);
  if (!match) return null;
  const [, , expiryStr, strikeStr, cOrP] = match;
  return { expiryStr, strike: Number(strikeStr), type: cOrP === 'C' ? 'call' : 'put' };
}

/** Fetch BTC or ETH options chain from Deribit */
async function fetchDeribitChain(currency: 'BTC' | 'ETH', expiryFilter?: string) {
  const url = `${DERIBIT_BASE}/get_book_summary_by_currency?currency=${currency}&kind=option`;
  const res = await fetch(url, { next: { revalidate: 30 }, signal: AbortSignal.timeout(15000) });
  if (!res.ok) throw new Error(`Deribit ${res.status}`);
  const json = await res.json();
  const list: DeribitBookSummary[] = json.result ?? [];
  let spot = 0;
  const byExpiry = new Map<string, { calls: OptionRow[]; puts: OptionRow[] }>();
  const expiriesSet = new Set<string>();

  for (const row of list) {
    const parsed = parseDeribitName(row.instrument_name);
    if (!parsed) continue;
    if (row.underlying_price) spot = row.underlying_price;
    expiriesSet.add(parsed.expiryStr);
    if (expiryFilter && parsed.expiryStr !== expiryFilter) continue;

    const optionRow: OptionRow = {
      strike: parsed.strike,
      bid: row.bid_price ?? 0,
      ask: row.ask_price ?? 0,
      mark: row.mark_price ?? (row.bid_price && row.ask_price ? (row.bid_price + row.ask_price) / 2 : 0),
      bidIv: row.mark_iv != null ? row.mark_iv : 0,
      askIv: row.mark_iv != null ? row.mark_iv : 0,
      delta: undefined,
      bidSize: undefined,
      askSize: undefined,
    };

    if (!byExpiry.has(parsed.expiryStr)) {
      byExpiry.set(parsed.expiryStr, { calls: [], puts: [] });
    }
    const bucket = byExpiry.get(parsed.expiryStr)!;
    if (parsed.type === 'call') bucket.calls.push(optionRow);
    else bucket.puts.push(optionRow);
  }

  const expiries = Array.from(expiriesSet).sort();
  const selectedExpiry = expiryFilter ?? expiries[0];
  const bucket = byExpiry.get(selectedExpiry) ?? { calls: [], puts: [] };
  const strikes = new Set<number>([...bucket.calls.map((c) => c.strike), ...bucket.puts.map((p) => p.strike)]);
  const strikesSorted = Array.from(strikes).sort((a, b) => a - b);

  const callsByStrike = new Map(bucket.calls.map((c) => [c.strike, c]));
  const putsByStrike = new Map(bucket.puts.map((p) => [p.strike, p]));
  const calls = strikesSorted.map((strike) => callsByStrike.get(strike) ?? { strike, bid: 0, ask: 0, mark: 0, bidIv: 0, askIv: 0, delta: undefined, bidSize: undefined, askSize: undefined });
  const puts = strikesSorted.map((strike) => putsByStrike.get(strike) ?? { strike, bid: 0, ask: 0, mark: 0, bidIv: 0, askIv: 0, delta: undefined, bidSize: undefined, askSize: undefined });

  return { spot, expiries, expiry: selectedExpiry, calls, puts };
}

interface OptionRow {
  strike: number;
  bid: number;
  ask: number;
  mark: number;
  bidIv: number;
  askIv: number;
  delta?: number;
  bidSize?: number;
  askSize?: number;
}

/** Fetch equity options chain from backend */
async function fetchEquityChain(symbol: string, expiry?: string) {
  const base = getBackendUrl();
  const q = expiry ? `?expiry_date=${expiry}` : '';
  const res = await fetch(`${base}/api/options/${symbol}/chain${q}`, { next: { revalidate: 60 }, signal: AbortSignal.timeout(15000) });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Backend ${res.status}`);
  }
  const data = await res.json();
  const callsRaw = data.calls ?? [];
  const putsRaw = data.puts ?? [];
  const toRow = (r: Record<string, unknown>): OptionRow => ({
    strike: Number(r.strike ?? 0),
    bid: Number(r.bid ?? 0),
    ask: Number(r.ask ?? 0),
    mark: Number(r.lastPrice ?? r.last_price ?? (Number(r.bid) + Number(r.ask)) / 2),
    bidIv: Number(r.impliedVolatility ?? r.implied_volatility ?? 0) * 100,
    askIv: Number(r.impliedVolatility ?? r.implied_volatility ?? 0) * 100,
    delta: r.delta != null ? Number(r.delta) : undefined,
    bidSize: r.bidSize != null ? Number(r.bidSize) : undefined,
    askSize: r.askSize != null ? Number(r.askSize) : undefined,
  });
  const calls = callsRaw.map(toRow).sort((a: OptionRow, b: OptionRow) => a.strike - b.strike);
  const puts = putsRaw.map(toRow).sort((a: OptionRow, b: OptionRow) => a.strike - b.strike);
  let spot = 0;
  try {
    const spotRes = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || ''}/api/real-market-data?symbols=${symbol}`, { cache: 'no-store' });
    const spotJson = await spotRes.json();
    const key = symbol === 'SPY' ? 'SPY' : symbol;
    spot = spotJson?.data?.[key]?.price ?? spotJson?.data?.[`${symbol}-USD`]?.price ?? 0;
  } catch {
    // ignore
  }
  return {
    spot,
    expiries: data.available_expiries ?? [data.expiry_date].filter(Boolean),
    expiry: data.expiry_date ?? expiry ?? '',
    calls,
    puts,
  };
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const symbol = (searchParams.get('symbol') || 'BTC').toUpperCase();
  const expiry = searchParams.get('expiry') || undefined;

  try {
    const isCrypto = symbol === 'BTC' || symbol === 'ETH';
    const result = isCrypto
      ? await fetchDeribitChain(symbol as 'BTC' | 'ETH', expiry)
      : await fetchEquityChain(symbol, expiry);

    return NextResponse.json({
      ok: true,
      symbol,
      spot: result.spot,
      expiries: result.expiries,
      expiry: result.expiry,
      calls: result.calls,
      puts: result.puts,
    });
  } catch (e) {
    const message = e instanceof Error ? e.message : 'Failed to fetch options chain';
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
