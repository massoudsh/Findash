'use client';

import { useEffect, useState, useCallback } from 'react';

export interface TickerItem {
  symbol: string;
  label: string;
  icon: string;
  category: string;
  price: number | null;
  change_pct: number | null;
  up: boolean | null;
  available: boolean;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8011';
const REFRESH_MS = 60_000;

export function useIranTicker() {
  const [items, setItems] = useState<TickerItem[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchTicker = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/iran-market/ticker`, { cache: 'no-store' });
      if (!res.ok) return;
      const data = await res.json();
      if (Array.isArray(data.items)) {
        setItems(data.items);
        setLoading(false);
      }
    } catch {
      // keep previous values on error
    }
  }, []);

  useEffect(() => {
    fetchTicker();
    const id = setInterval(fetchTicker, REFRESH_MS);
    return () => clearInterval(id);
  }, [fetchTicker]);

  return { items, loading };
}
