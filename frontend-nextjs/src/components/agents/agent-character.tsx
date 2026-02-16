'use client';

import {
  Activity,
  Archive,
  BarChart3,
  Brain,
  Copy,
  Database,
  History,
  MessageSquare,
  Shield,
  Target,
  TrendingUp,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  type AgentId,
  getAgentCharacter,
  AGENT_CHARACTERS,
} from '@/lib/agent-characters';

const ICON_MAP = {
  Activity,
  Archive,
  BarChart3,
  Brain,
  Copy,
  Database,
  History,
  MessageSquare,
  Shield,
  Target,
  TrendingUp,
} as const;

interface AgentCharacterProps {
  agentId: AgentId;
  /** 'avatar' = emoji circle only; 'inline' = avatar + shortName + tagline; 'full' = avatar + name + tagline */
  variant?: 'avatar' | 'inline' | 'full';
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export function AgentCharacter({
  agentId,
  variant = 'inline',
  className,
  size = 'md',
}: AgentCharacterProps) {
  const agent = getAgentCharacter(agentId);
  const Icon = ICON_MAP[agent.icon as keyof typeof ICON_MAP] ?? BarChart3;

  const sizeClasses = {
    avatar: { sm: 'h-8 w-8 text-sm', md: 'h-10 w-10 text-base', lg: 'h-12 w-12 text-lg' },
    icon: { sm: 'h-3.5 w-3.5', md: 'h-4 w-4', lg: 'h-5 w-5' },
  };

  const avatar = (
    <div
      className={cn(
        'flex shrink-0 items-center justify-center rounded-xl ring-2 backdrop-blur-sm',
        agent.colorClass,
        sizeClasses.avatar[size]
      )}
      title={`${agent.name} (${agent.shortName})`}
    >
      <Icon className={cn('shrink-0', sizeClasses.icon[size])} />
    </div>
  );

  if (variant === 'avatar') {
    return <div className={cn('flex', className)}>{avatar}</div>;
  }

  const displayName = variant === 'full' ? agent.name : agent.shortName;
  return (
    <div className={cn('flex items-center gap-2 min-w-0', className)}>
      {avatar}
      <div className="min-w-0 flex flex-col">
        <span className="text-sm font-semibold text-foreground truncate">
          {displayName}
        </span>
        <span className="text-xs text-muted-foreground truncate" title={agent.tagline}>
          {agent.tagline}
        </span>
      </div>
    </div>
  );
}

/** Compact badge showing agent id + shortName for lists/filters */
export function AgentBadge({ agentId, className }: { agentId: AgentId; className?: string }) {
  const agent = getAgentCharacter(agentId);
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium',
        'bg-muted/80 text-muted-foreground',
        className
      )}
    >
      <span className="opacity-80">{agent.id}</span>
      <span>{agent.shortName}</span>
    </span>
  );
}

export { AGENT_CHARACTERS, getAgentCharacter };
export type { AgentId, AgentCharacter };
