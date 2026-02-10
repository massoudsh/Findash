import { Suspense } from 'react';
import { StrategiesContent } from '@/components/strategies/strategies-content';

export default function StrategiesPage() {
  return (
    <div className="container mx-auto px-6 py-8">
      <Suspense fallback={<div className="text-center">Loading strategies...</div>}>
        <StrategiesContent />
      </Suspense>
    </div>
  );
} 