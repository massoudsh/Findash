'use client';

import { useRef } from 'react';
import Link from 'next/link';
import { motion, useInView } from 'framer-motion';
import { Database, Brain, User, Zap, ArrowRight } from 'lucide-react';

const PHASES = [
  {
    id: 'sources',
    label: 'Sources',
    description: 'Markets, news, alternative data.',
    icon: Database,
    color: 'bg-blue-500/15 text-blue-600 dark:text-blue-400 border-blue-500/30',
  },
  {
    id: 'analyze',
    label: 'Analyze',
    description: 'Signals, risk, and forecasts.',
    icon: Brain,
    color: 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30',
  },
  {
    id: 'decide',
    label: 'Decide',
    description: 'You review and take the call.',
    icon: User,
    color: 'bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30',
  },
  {
    id: 'execute',
    label: 'Execute',
    description: 'Orders, backtest, reports.',
    icon: Zap,
    color: 'bg-violet-500/15 text-violet-600 dark:text-violet-400 border-violet-500/30',
  },
];

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.12 },
  },
};

const cardItem = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
};

export function PipelinePhases() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });

  return (
    <motion.div
      ref={ref}
      variants={container}
      initial="hidden"
      animate={inView ? 'show' : 'hidden'}
      className="grid gap-4 grid-cols-1 min-[500px]:grid-cols-2 lg:grid-cols-4"
    >
      {PHASES.map((phase) => {
        const Icon = phase.icon;
        return (
          <motion.div key={phase.id} variants={cardItem}>
            <Link
              href="/workflow"
              className={`group flex flex-col rounded-xl border p-4 sm:p-5 transition-all duration-300 hover:shadow-lg hover:shadow-black/10 hover:-translate-y-0.5 hover:border-primary/30 dark:hover:shadow-black/20 ${phase.color}`}
            >
              <div className="mb-3 flex items-center justify-between">
                <span className="rounded-lg bg-background/50 p-2">
                  <Icon className="h-5 w-5" aria-hidden />
                </span>
                <ArrowRight className="h-4 w-4 opacity-0 transition-opacity group-hover:opacity-100" aria-hidden />
              </div>
              <h3 className="font-semibold text-foreground">{phase.label}</h3>
              <p className="mt-1 text-sm text-muted-foreground line-clamp-2">{phase.description}</p>
            </Link>
          </motion.div>
        );
      })}
    </motion.div>
  );
}
