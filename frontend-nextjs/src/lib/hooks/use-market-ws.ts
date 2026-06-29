'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

export interface MarketTick {
  symbol: string;
  price: number;
  change: number;
  changePercent?: number;
  volume: number;
  timestamp: string;
  prev?: number; // for flash animation
}

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'polling';

const POLL_INTERVAL = 4000;
const WS_URL =
  typeof window !== 'undefined'
    ? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname}:8000/ws/market-stream`
    : null;

/**
 * useMarketWS — connects to backend WebSocket for live prices.
 * Falls back to polling /api/real-market-data every 4s if WS is unavailable.
 */
export function useMarketWS(symbols: string[]) {
  const [ticks, setTicks] = useState<Record<string, MarketTick>>({});
  const [status, setStatus] = useState<ConnectionStatus>('connecting');
  const wsRef = useRef<WebSocket | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const retriesRef = useRef(0);
  const symbolsRef = useRef(symbols);
  symbolsRef.current = symbols;

  const mergeTick = useCallback((next: MarketTick) => {
    setTicks((prev) => {
      const old = prev[next.symbol];
      return {
        ...prev,
        [next.symbol]: {
          ...next,
          prev: old?.price,
        },
      };
    });
  }, []);

  // ── Polling fallback ──────────────────────────────────────────────────────
  const startPolling = useCallback(() => {
    setStatus('polling');
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);

    const fetch_data = async () => {
      try {
        const syms = symbolsRef.current.join(',');
        const res = await fetch(`/api/real-market-data?symbols=${syms}`);
        if (!res.ok) return;
        const json = await res.json();
        if (!json.data) return;
        Object.values(json.data as Record<string, any>).forEach((d: any) => {
          mergeTick({
            symbol: d.symbol,
            price: d.price,
            change: d.change,
            changePercent: d.change_percent,
            volume: d.volume,
            timestamp: d.timestamp || new Date().toISOString(),
          });
        });
      } catch {
        // silent
      }
    };

    fetch_data();
    pollTimerRef.current = setInterval(fetch_data, POLL_INTERVAL);
  }, [mergeTick]);

  // ── WebSocket ─────────────────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    if (!WS_URL) {
      startPolling();
      return;
    }
    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      setStatus('connecting');

      const connectTimeout = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          ws.close();
          startPolling();
        }
      }, 3000);

      ws.onopen = () => {
        clearTimeout(connectTimeout);
        setStatus('connected');
        retriesRef.current = 0;
        ws.send(JSON.stringify({ type: 'subscribe', symbols: symbolsRef.current }));
      };

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'tick' && msg.data) {
            mergeTick(msg.data);
          } else if (msg.type === 'batch' && Array.isArray(msg.data)) {
            msg.data.forEach((d: any) => mergeTick(d));
          }
        } catch {
          // ignore parse errors
        }
      };

      ws.onerror = () => {
        clearTimeout(connectTimeout);
      };

      ws.onclose = () => {
        clearTimeout(connectTimeout);
        setStatus('disconnected');
        // Fallback to polling after 2 failed WS attempts
        if (retriesRef.current >= 2) {
          startPolling();
        } else {
          retriesRef.current += 1;
          setTimeout(connectWS, 2000);
        }
      };
    } catch {
      startPolling();
    }
  }, [startPolling, mergeTick]);

  useEffect(() => {
    connectWS();
    return () => {
      wsRef.current?.close();
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return { ticks, status };
}
