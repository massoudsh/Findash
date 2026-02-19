'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { ArrowRight, Play } from 'lucide-react';

export function HeroSection() {
  return (
    <section className="relative overflow-hidden rounded-xl sm:rounded-2xl border-2 border-amber-400/60 bg-gradient-to-br from-amber-50/90 to-yellow-100/80 dark:from-amber-950/50 dark:to-yellow-900/20 py-10 sm:py-16 md:py-20 px-4 sm:px-8 md:px-10">
      {/* Subtle animated background: floating data nodes */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden>
        {[...Array(12)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 rounded-full bg-amber-400/20 dark:bg-amber-500/10"
            style={{
              left: `${10 + (i * 7) % 80}%`,
              top: `${15 + (i * 11) % 70}%`,
            }}
            animate={{
              y: [0, -8, 0],
              opacity: [0.3, 0.6, 0.3],
            }}
            transition={{
              duration: 3 + (i % 3),
              repeat: Infinity,
              delay: i * 0.2,
            }}
          />
        ))}
        <motion.div
          className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-amber-400/30 to-transparent"
          animate={{ opacity: [0.4, 0.8, 0.4] }}
          transition={{ duration: 4, repeat: Infinity }}
        />
      </div>

      <div className="relative z-10 text-center max-w-3xl mx-auto">
        <motion.h1
          className="text-2xl font-bold tracking-tight text-foreground min-[480px]:text-3xl sm:text-4xl md:text-5xl"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Decisive AI execution for better returns, less cognitive load.
        </motion.h1>
        <motion.p
          className="mt-3 sm:mt-4 text-muted-foreground text-base sm:text-lg md:text-xl max-w-2xl mx-auto"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          Data, signals, and risk in one place. You decide; agents assist.
        </motion.p>
        <motion.div
          className="mt-6 sm:mt-8 flex flex-col min-[400px]:flex-row flex-wrap items-stretch min-[400px]:items-center justify-center gap-3 sm:gap-4"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Button
            asChild
            size="default"
            className="bg-amber-500 hover:bg-amber-600 text-amber-950 font-semibold shadow-md border-0 w-full min-[400px]:w-auto h-10 sm:h-11 px-4 sm:px-6"
          >
            <Link href="/trading" className="justify-center">
              Open Command Center
              <ArrowRight className="ml-2 h-4 w-4 shrink-0" />
            </Link>
          </Button>
          <Button
            asChild
            size="default"
            variant="outline"
            className="border-amber-500/60 text-amber-700 dark:text-amber-300 hover:bg-amber-500/10 w-full min-[400px]:w-auto h-10 sm:h-11 px-4 sm:px-6"
          >
            <Link href="/workflow" className="justify-center">
              <Play className="mr-2 h-4 w-4 shrink-0" />
              See how it works
            </Link>
          </Button>
        </motion.div>
      </div>
    </section>
  );
}
