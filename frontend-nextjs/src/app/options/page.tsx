'use client';

import { Suspense } from 'react';
import { OptionsPageContent } from './options-page-content';

export default function OptionsPage() {
  return (
    <div className="w-full h-full min-h-screen flex flex-col">
      <Suspense fallback={<div className="flex flex-1 items-center justify-center text-muted-foreground">Loading options…</div>}>
        <OptionsPageContent />
      </Suspense>
    </div>
  );
}
