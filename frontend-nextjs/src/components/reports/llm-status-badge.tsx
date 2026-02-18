'use client';

import { useEffect, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Sparkles } from 'lucide-react';

interface LlmStatus {
  any_configured: boolean;
  falcon_configured?: boolean;
  fingpt_local_configured?: boolean;
  hf_configured?: boolean;
}

export function LlmStatusBadge() {
  const [status, setStatus] = useState<LlmStatus | null>(null);

  useEffect(() => {
    fetch('/api/llm/status')
      .then((res) => res.json())
      .then(setStatus)
      .catch(() => setStatus({ any_configured: false }));
  }, []);

  if (status === null) return null;

  if (status.any_configured) {
    return (
      <Badge variant="outline" className="text-xs border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300">
        <Sparkles className="h-3 w-3 mr-1" />
        LLM ready
      </Badge>
    );
  }

  return (
    <Badge variant="outline" className="text-xs text-muted-foreground">
      <Sparkles className="h-3 w-3 mr-1" />
      Simulated
    </Badge>
  );
}
