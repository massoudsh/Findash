import { Suspense } from 'react';
import { ReportsContent } from '@/components/reports/reports-content';

export default function ReportsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">AI-Powered Reports</h1>
        <p className="text-muted-foreground">
          Intelligent insights and comprehensive analysis powered by Llama AI models
        </p>
      </div>
      <Suspense fallback={<div>Loading AI reporting system...</div>}>
        <ReportsContent />
      </Suspense>
    </div>
  );
} 