import { Suspense } from 'react';
import { SocialContent } from '@/components/social/social-content';
import { SentimentAgentPanel } from '@/components/trading/sentiment-agent-panel';

export default function SocialPage() {
  return (
    <div className="container mx-auto px-6 py-8">
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
        <div className="min-w-0">
          <Suspense fallback={<div className="text-center">Loading social signals...</div>}>
            <SocialContent />
          </Suspense>
        </div>
        <aside className="hidden xl:block min-h-[360px]">
          <SentimentAgentPanel />
        </aside>
      </div>
    </div>
  );
}
