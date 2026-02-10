import { Suspense } from 'react';
import { RealtimeContent } from '@/components/realtime/realtime-content';

export default function RealtimePage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Real-time Processing</h1>
        <p className="text-muted-foreground">
          Monitor live market data streams and real-time analytics
        </p>
      </div>
      <Suspense fallback={<div>Loading real-time data...</div>}>
        <RealtimeContent />
      </Suspense>
    </div>
  );
} 