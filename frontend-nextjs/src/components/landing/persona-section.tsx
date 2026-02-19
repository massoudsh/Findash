'use client';

import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import { User, FlaskConical, Building2 } from 'lucide-react';

const PERSONAS = [
  {
    icon: User,
    label: 'Retail trader',
    problem: 'Too much noise, not enough signal.',
    solution: 'One command center: data, signals, risk, execute.',
  },
  {
    icon: FlaskConical,
    label: 'Quant researcher',
    problem: 'Backtests and live execution disconnected.',
    solution: 'Unified pipeline from backtest to paper to live.',
  },
  {
    icon: Building2,
    label: 'Fund manager',
    problem: 'Need oversight without slowing execution.',
    solution: 'Full transparency and control with agent-assisted flow.',
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
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0 },
};

export function PersonaSection() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });

  return (
    <motion.section
      ref={ref}
      variants={container}
      initial="hidden"
      animate={inView ? 'show' : 'hidden'}
      className="rounded-xl sm:rounded-2xl border border-border/60 bg-card/50 p-6 sm:p-8 lg:p-10"
    >
      <h2 className="text-xl font-semibold text-foreground mb-2">Built For</h2>
      <p className="text-sm text-muted-foreground mb-8 max-w-xl">
        One system that adapts to how you work.
      </p>
      <div className="grid gap-4 sm:gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
        {PERSONAS.map((p) => {
          const Icon = p.icon;
          return (
            <motion.div
              key={p.label}
              variants={item}
              className="rounded-lg sm:rounded-xl border border-border/40 bg-background/40 p-4 sm:p-5 transition-all hover:border-primary/20 hover:shadow-md"
            >
              <div className="mb-3 rounded-lg bg-primary/10 p-2 w-fit">
                <Icon className="h-5 w-5 text-primary" aria-hidden />
              </div>
              <h3 className="font-semibold text-foreground">{p.label}</h3>
              <p className="mt-2 text-xs text-muted-foreground line-clamp-2">
                <span className="font-medium text-foreground">Problem:</span> {p.problem}
              </p>
              <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
                <span className="font-medium text-foreground">Solution:</span> {p.solution}
              </p>
            </motion.div>
          );
        })}
      </div>
    </motion.section>
  );
}
