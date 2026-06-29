'use client';

import { useState, useEffect, useRef, useCallback } from 'react';

export interface PriceAlert {
  id: string;
  symbol: string;
  targetPrice: number;
  direction: 'above' | 'below';
  note?: string;
  createdAt: string;
  triggered?: boolean;
  triggeredAt?: string;
}

const STORAGE_KEY = 'octopus_price_alerts_v1';

function load(): PriceAlert[] {
  if (typeof window === 'undefined') return [];
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
}

function save(alerts: PriceAlert[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(alerts));
}

/**
 * usePriceAlerts — stores price alerts in localStorage.
 * Pass current prices and an onTrigger callback to get notified when a price crosses a threshold.
 */
export function usePriceAlerts(
  prices: Record<string, number>,
  onTrigger: (alert: PriceAlert, price: number) => void
) {
  const [alerts, setAlerts] = useState<PriceAlert[]>([]);
  const triggeredIds = useRef<Set<string>>(new Set());

  useEffect(() => {
    const stored = load();
    setAlerts(stored);
    stored.filter((a) => a.triggered).forEach((a) => triggeredIds.current.add(a.id));
  }, []);

  // Check prices against active alerts
  useEffect(() => {
    if (Object.keys(prices).length === 0) return;
    setAlerts((prev) => {
      let changed = false;
      const next = prev.map((alert) => {
        if (alert.triggered) return alert;
        const price = prices[alert.symbol];
        if (price === undefined) return alert;
        const hit =
          (alert.direction === 'above' && price >= alert.targetPrice) ||
          (alert.direction === 'below' && price <= alert.targetPrice);
        if (hit && !triggeredIds.current.has(alert.id)) {
          triggeredIds.current.add(alert.id);
          onTrigger(alert, price);
          changed = true;
          return { ...alert, triggered: true, triggeredAt: new Date().toISOString() };
        }
        return alert;
      });
      if (changed) save(next);
      return changed ? next : prev;
    });
  }, [prices, onTrigger]);

  const addAlert = useCallback((data: Omit<PriceAlert, 'id' | 'createdAt' | 'triggered'>) => {
    const alert: PriceAlert = {
      ...data,
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      triggered: false,
    };
    setAlerts((prev) => {
      const next = [alert, ...prev];
      save(next);
      return next;
    });
  }, []);

  const removeAlert = useCallback((id: string) => {
    triggeredIds.current.delete(id);
    setAlerts((prev) => {
      const next = prev.filter((a) => a.id !== id);
      save(next);
      return next;
    });
  }, []);

  const clearTriggered = useCallback(() => {
    setAlerts((prev) => {
      const next = prev.filter((a) => !a.triggered);
      save(next);
      return next;
    });
  }, []);

  return { alerts, addAlert, removeAlert, clearTriggered };
}
