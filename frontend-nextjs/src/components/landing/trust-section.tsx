'use client';

import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import { Lock, Database, Cpu, BarChart3, Shield, Server, Wallet } from 'lucide-react';

const TRUST_ITEMS = [
  {
    icon: Database,
    title: 'Data source transparency',
    line: 'Know where every feed comes from and how it’s normalized.',
  },
  {
    icon: Lock,
    title: 'Security first',
    line: 'Encrypted in transit and at rest. No custody of your funds.',
  },
  {
    icon: Cpu,
    title: 'AI control transparency',
    line: 'You see agent logic and overrides. No hidden automation.',
  },
  {
    icon: BarChart3,
    title: 'Performance metrics',
    line: 'Latency, fill rates, and strategy stats in one place.',
  },
];

const BADGES = [
  { icon: Lock, label: 'Encrypted' },
  { icon: Server, label: 'Self-hosted' },
  { icon: Wallet, label: 'No custody' },
];

export function TrustSection() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });

  return (
    <motion.section
      ref={ref}
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5 }}
      className="rounded-xl sm:rounded-2xl border border-border/60 bg-card/50 p-6 sm:p-8 lg:p-10"
    >
      <h2 className="text-xl font-semibold text-foreground mb-2">Why Trust Nexus?</h2>
      <p className="text-sm text-muted-foreground mb-8 max-w-xl">
        Built for clarity and control: transparent data, strong security, and you in the loop.
      </p>
      <div className="grid gap-4 sm:gap-6 grid-cols-1 min-[500px]:grid-cols-2 lg:grid-cols-4 mb-6 sm:mb-8">
        {TRUST_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <div
              key={item.title}
              className="rounded-lg sm:rounded-xl border border-border/40 bg-background/40 p-3 sm:p-4 transition-colors hover:border-primary/20"
            >
              <Icon className="h-5 w-5 text-primary mb-2" aria-hidden />
              <h3 className="font-medium text-foreground text-sm mb-1">{item.title}</h3>
              <p className="text-xs text-muted-foreground line-clamp-2">{item.line}</p>
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-3">
        {BADGES.map((b) => {
          const Icon = b.icon;
          return (
            <span
              key={b.label}
              className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-muted/30 px-4 py-2 text-xs font-medium text-foreground"
            >
              <Icon className="h-3.5 w-3.5" aria-hidden />
              {b.label}
            </span>
          );
        })}
      </div>
    </motion.section>
  );
}
