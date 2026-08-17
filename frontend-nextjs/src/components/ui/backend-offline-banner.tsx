'use client';

import { WifiOff, RefreshCw } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface BackendOfflineBannerProps {
  backendUrl: string;
  message: string;
  fallbackLabel: string;
  onRetry: () => void;
}

/** Shown when the FastAPI backend is unreachable, so the UI can fall back to local/sample data. */
export function BackendOfflineBanner({ backendUrl, message, fallbackLabel, onRetry }: BackendOfflineBannerProps) {
  return (
    <Card className="border-amber-500/30 bg-amber-500/5">
      <CardContent className="py-3 px-4 flex items-center gap-3 flex-wrap">
        <WifiOff className="h-5 w-5 text-amber-500 shrink-0" />
        <div className="text-sm flex-1 min-w-[200px]">
          <span className="font-medium">{message}</span>{' '}
          <span className="text-muted-foreground" dir="ltr">
            {backendUrl}
          </span>
          <span className="block text-muted-foreground mt-0.5">{fallbackLabel}</span>
        </div>
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RefreshCw className="h-4 w-4 mr-2" />
          تلاش مجدد
        </Button>
      </CardContent>
    </Card>
  );
}
