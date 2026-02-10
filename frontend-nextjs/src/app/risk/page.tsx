import { Suspense } from 'react';
import { RiskContent } from '@/components/risk/risk-content';

export default function RiskPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Risk Dashboard</h1>
        <p className="text-muted-foreground">
          Analyze portfolio risk and run stress tests.
        </p>
      </div>
      <Suspense fallback={<div>Loading risk analysis...</div>}>
        <RiskContent />
      </Suspense>
    </div>
  );
} 