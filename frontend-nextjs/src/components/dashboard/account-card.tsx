'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { formatCurrency } from '@/lib/utils';
import { Eye, MoreHorizontal, Plus, Minus, CreditCard, Wallet, Landmark } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface AccountCardData {
  id: string;
  name: string;
  number: string;
  balance: number;
  type: 'visa' | 'crypto' | 'savings';
  gradient: string;
}

const typeConfig = {
  visa: {
    icon: CreditCard,
    label: 'Trading',
    brand: 'VISA',
    glassTint: 'from-emerald-500/15 to-teal-600/10',
  },
  crypto: {
    icon: Wallet,
    label: 'Crypto',
    brand: '••••',
    glassTint: 'from-cyan-500/15 to-emerald-600/10',
  },
  savings: {
    icon: Landmark,
    label: 'Savings',
    brand: '••••',
    glassTint: 'from-green-500/15 to-emerald-700/10',
  },
};

function CardChip() {
  return (
    <div
      className="absolute left-5 top-9 w-9 h-7 rounded bg-gradient-to-br from-amber-200/80 to-amber-400/70 shadow-inner border border-amber-500/20"
      aria-hidden
    >
      <div className="absolute inset-0.5 flex flex-wrap gap-px">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-1 w-1.5 bg-amber-600/40 rounded-sm" />
        ))}
      </div>
    </div>
  );
}

export function AccountCard({
  account,
  className,
}: {
  account: AccountCardData;
  className?: string;
}) {
  const config = typeConfig[account.type];
  const Icon = config.icon;

  return (
    <Card
      variant="default"
      hover={true}
      className={cn(
        'group/card overflow-hidden rounded-2xl transition-all duration-300 flex flex-col w-full',
        'border border-white/20 dark:border-white/10',
        'bg-white/15 dark:bg-white/10 backdrop-blur-xl',
        'shadow-[0_8px_32px_rgba(0,0,0,0.08)] dark:shadow-[0_8px_32px_rgba(0,0,0,0.2)]',
        'hover:shadow-[0_12px_40px_rgba(0,0,0,0.12)] hover:border-emerald-400/30 dark:hover:border-emerald-500/20',
        'hover:bg-white/20 dark:hover:bg-white/15',
        'min-h-[200px]',
        className
      )}
    >
      {/* Glass tint overlay (greeny fintech) */}
      <div
        className={cn(
          'absolute inset-0 pointer-events-none rounded-2xl bg-gradient-to-br opacity-90',
          config.glassTint
        )}
        aria-hidden
      />

      {/* Shine on hover */}
      <div
        className="absolute inset-0 opacity-0 group-hover/card:opacity-100 transition-opacity duration-500 pointer-events-none rounded-2xl"
        style={{
          background:
            'linear-gradient(135deg, rgba(255,255,255,0.2) 0%, transparent 50%)',
        }}
        aria-hidden
      />

      <CardContent className="relative flex flex-1 flex-col p-5 text-foreground overflow-hidden">
        {/* Top row: icon + label + actions */}
        <div className="flex items-start justify-between gap-3 mb-4">
          <div className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/20 dark:bg-emerald-500/25 border border-emerald-400/20 backdrop-blur-sm">
              <Icon className="h-5 w-5 text-emerald-700 dark:text-emerald-300" />
            </div>
            <span className="text-xs font-semibold tracking-widest uppercase text-muted-foreground">
              {config.label}
            </span>
          </div>
          <div className="flex items-center gap-0.5">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/20 border-0"
              aria-label="View details"
            >
              <Eye className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/20 border-0"
              aria-label="More options"
            >
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <CardChip />

        {/* Card number */}
        <div className="mt-8 mb-3">
          <p
            className="font-mono text-sm tracking-[0.25em] text-foreground/90 tabular-nums"
            style={{ letterSpacing: '0.22em' }}
          >
            {account.number}
          </p>
        </div>

        {/* Bottom: name + balance + actions */}
        <div className="mt-auto flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-0.5">
              {account.name}
            </p>
            <p className="text-xl font-bold tracking-tight tabular-nums text-foreground">
              {formatCurrency(account.balance)}
            </p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <Button
              variant="ghost"
              size="sm"
              className="h-8 rounded-lg text-muted-foreground hover:text-foreground hover:bg-emerald-500/20 text-xs font-medium border border-emerald-500/20"
            >
              <Plus className="h-3.5 w-3.5 mr-1" />
              Deposit
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 rounded-lg text-muted-foreground hover:text-foreground hover:bg-emerald-500/20 text-xs font-medium border border-emerald-500/20"
            >
              <Minus className="h-3.5 w-3.5 mr-1" />
              Withdraw
            </Button>
          </div>
        </div>

        {/* Brand */}
        <div className="absolute right-4 bottom-4">
          <span className="text-[10px] font-bold tracking-widest text-muted-foreground/80 uppercase">
            {config.brand}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
