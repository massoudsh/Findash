'use client';

import { useEffect, useState, useCallback } from 'react';

const HEALTH_TIMEOUT_MS = 4_000;

export interface BackendHealthState {
  ok: boolean;
  backendUrl: string;
  loading: boolean;
  refetch: () => void;
}

/** Checks `${NEXT_PUBLIC_API_URL}/health` to detect whether the backend is reachable. */
export function useBackendHealth(): BackendHealthState {
  const backendUrl =
    (typeof process !== 'undefined' ? process.env.NEXT_PUBLIC_API_URL : '') || 'http://localhost:8000';
  const [ok, setOk] = useState(false);
  const [loading, setLoading] = useState(true);

  const check = useCallback(async () => {
    setLoading(true);
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
      const res = await fetch(`${backendUrl}/health`, { cache: 'no-store', signal: controller.signal });
      clearTimeout(timeout);
      setOk(res.ok);
    } catch {
      setOk(false);
    } finally {
      setLoading(false);
    }
  }, [backendUrl]);

  useEffect(() => {
    check();
  }, [check]);

  return { ok, backendUrl, loading, refetch: check };
}
