import { Suspense } from 'react';
import { PortfolioContent } from '@/components/portfolio/portfolio-content';

export default function PortfolioPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Portfolio</h1>
        <p className="text-muted-foreground">
          Track your investments and asset allocation
        </p>
      </div>
      <Suspense fallback={<div>Loading portfolio...</div>}>
        <PortfolioContent />
      </Suspense>
    </div>
  );
} 