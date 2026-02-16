'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from '@/lib/utils';

/**
 * Reusable agent panel for platform agents (M1 Data Collector, M4 Strategy, M9 Sentiment, M11 Analysis).
 * Keeps UI and behavior consistent across Trading Center and other sections.
 */
interface AgentPanelProps {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  defaultCollapsed?: boolean;
}

export function AgentPanel({
  title,
  subtitle,
  icon,
  children,
  className,
  defaultCollapsed = false,
}: AgentPanelProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  return (
    <Card className={cn('flex flex-col h-full min-h-0 border bg-card/95', className)}>
      <CardHeader className="shrink-0 py-3 px-4 border-b">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            {icon}
            {title}
          </CardTitle>
          <button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            className="p-1 rounded hover:bg-muted"
            aria-label={collapsed ? 'Expand' : 'Collapse'}
          >
            {collapsed ? (
              <ChevronUp className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            )}
          </button>
        </div>
        <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>
      </CardHeader>
      {!collapsed && (
        <CardContent className="p-0 flex-1 min-h-0 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto px-3 py-2">{children}</div>
        </CardContent>
      )}
    </Card>
  );
}
