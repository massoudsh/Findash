import { Suspense } from 'react';
import { SocialContent } from '@/components/social/social-content';
import { SentimentAgentPanel } from '@/components/trading/sentiment-agent-panel';

export default function SocialPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Social Signals</h1>
        <p className="text-muted-foreground mt-1">
          Sentiment from Twitter, Reddit, Fear & Greed, and news — real-time when backend is available
        </p>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
        <div className="min-w-0">
          <Suspense fallback={<div className="text-center text-muted-foreground py-8">Loading social signals...</div>}>
            <SocialContent />
          </Suspense>
        </div>
        <aside className="hidden xl:block xl:sticky xl:top-6 h-[calc(100vh-14rem)] min-h-[420px] max-h-[560px] overflow-y-auto">
          <SentimentAgentPanel />
        </aside>
      </div>
    </div>
  );
}
