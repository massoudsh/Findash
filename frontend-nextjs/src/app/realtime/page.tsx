import { Suspense } from 'react';
import { RealtimeContent } from '@/components/realtime/realtime-content';
import { RealtimeAgentPanel } from '@/components/agents/realtime-agent-panel';

export default function RealtimePage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">پردازش بلادرنگ</h1>
        <p className="text-muted-foreground">
          پایش جریان داده زنده بازار و تحلیل‌های بلادرنگ
        </p>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
        <div className="min-w-0">
          <Suspense fallback={<div>در حال بارگذاری داده‌های لحظه‌ای...</div>}>
            <RealtimeContent />
          </Suspense>
        </div>
        <aside className="hidden xl:block min-h-[360px]">
          <RealtimeAgentPanel />
        </aside>
      </div>
    </div>
  );
} 