'use client';

import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import { Database, Brain, Shield, TrendingUp } from 'lucide-react';

const BLOCKS = [
  {
    icon: Database,
    title: 'Data in one place',
    value: 'Markets, news, and signals unified so you see the full picture.',
    flow: ['Sources', 'Ingest', 'Stream'],
    color: 'from-blue-500/20 to-blue-600/5 border-blue-500/30',
    iconColor: 'text-blue-500',
  },
  {
    icon: Brain,
    title: 'AI that assists',
    value: 'Signals and risk prepared by agents; you keep the final say.',
    flow: ['Analyze', 'Signals', 'Risk'],
    color: 'from-emerald-500/20 to-emerald-600/5 border-emerald-500/30',
    iconColor: 'text-emerald-500',
  },
  {
    icon: Shield,
    title: 'You stay in control',
    value: 'No black boxes. Review, approve, or modify every step.',
    flow: ['Decide', 'Approve', 'Execute'],
    color: 'from-amber-500/20 to-amber-600/5 border-amber-500/30',
    iconColor: 'text-amber-500',
  },
  {
    icon: TrendingUp,
    title: 'Execution and reports',
    value: 'Paper or live execution, backtest, and clear reporting.',
    flow: ['Execute', 'Report', 'Insight'],
    color: 'from-violet-500/20 to-violet-600/5 border-violet-500/30',
    iconColor: 'text-violet-500',
  },
];

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const item = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0 },
};

export function InfographicTips() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });

  return (
    <motion.div
      ref={ref}
      variants={container}
      initial="hidden"
      animate={inView ? 'show' : 'hidden'}
      className="grid gap-4 sm:gap-6 grid-cols-1 min-[500px]:grid-cols-2 lg:grid-cols-4"
    >
      {BLOCKS.map((block) => {
        const Icon = block.icon;
        return (
          <motion.article
            key={block.title}
            variants={item}
            className={`flex min-h-0 flex-col rounded-xl border bg-gradient-to-br ${block.color} p-4 sm:p-5 transition-all duration-300 hover:shadow-lg hover:shadow-black/10 hover:scale-[1.02] dark:hover:shadow-black/20`}
          >
            <div className={`mb-3 rounded-lg bg-background/50 p-2.5 w-fit ${block.iconColor}`}>
              <Icon className="h-6 w-6" aria-hidden />
            </div>
            <h3 className="font-semibold text-foreground mb-1 line-clamp-1">{block.title}</h3>
            <p className="text-sm text-muted-foreground mb-4 line-clamp-2 flex-1 min-h-0">
              {block.value}
            </p>
            {/* Mini flow diagram */}
            <div className="flex items-center gap-1.5 flex-wrap">
              {block.flow.map((step, i) => (
                <span key={step} className="flex items-center gap-1.5">
                  {i > 0 && (
                    <span className="text-muted-foreground/50 text-xs" aria-hidden>
                      →
                    </span>
                  )}
                  <span className="rounded-md bg-background/60 px-2 py-1 text-xs font-medium text-foreground">
                    {step}
                  </span>
                </span>
              ))}
            </div>
          </motion.article>
        );
      })}
    </motion.div>
  );
}
