'use client';

import { useState } from 'react';
import { HeroSection } from '@/components/landing/hero-section';
import { InfographicTips } from '@/components/landing/infographic-tips';
import { PipelinePhases } from '@/components/landing/pipeline-phases';
import { TrustSection } from '@/components/landing/trust-section';
import { PersonaSection } from '@/components/landing/persona-section';
import {
  StickyCta,
  EmailCaptureBlock,
  OnboardingPreviewModal,
} from '@/components/landing/landing-conversion';
import { Button } from '@/components/ui/button';
import { Sparkles } from 'lucide-react';

const SECTION_PADDING = 'py-12 sm:py-16 md:py-20';
const CONTAINER = 'container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl w-full';

export default function HomePage() {
  const [onboardingOpen, setOnboardingOpen] = useState(false);

  return (
    <div className="min-h-[calc(100vh-6rem)] sm:min-h-[calc(100vh-8rem)]">
      {/* Hero */}
      <section className="pt-6 sm:pt-10 md:pt-12 pb-10 sm:pb-14 md:pb-16">
        <div className={CONTAINER}>
          <HeroSection />
        </div>
      </section>

      {/* Infographic tips (replaces text-heavy cards) */}
      <section className={SECTION_PADDING}>
        <div className={CONTAINER}>
          <h2 className="text-lg sm:text-xl font-semibold text-foreground mb-2 text-center sm:text-left">
            One system. Full control.
          </h2>
          <p className="text-sm text-muted-foreground mb-8 sm:mb-10 max-w-xl text-center sm:text-left line-clamp-2">
            From data to execution with clear steps and no black boxes.
          </p>
          <InfographicTips />
        </div>
      </section>

      {/* 4-phase pipeline (no Mermaid on landing) */}
      <section className={SECTION_PADDING}>
        <div className={CONTAINER}>
          <h2 className="text-lg sm:text-xl font-semibold text-foreground mb-2">How it works</h2>
          <p className="text-sm text-muted-foreground mb-8 sm:mb-10 max-w-xl line-clamp-2">
            Sources → Analyze → Decide → Execute. Simple pipeline, full transparency.
          </p>
          <PipelinePhases />
        </div>
      </section>

      {/* Trust */}
      <section className={SECTION_PADDING}>
        <div className={CONTAINER}>
          <TrustSection />
        </div>
      </section>

      {/* Persona — Built For */}
      <section className={SECTION_PADDING}>
        <div className={CONTAINER}>
          <PersonaSection />
        </div>
      </section>

      {/* Email capture */}
      <EmailCaptureBlock />

      {/* Quick start preview link + modal */}
      <div className={`${CONTAINER} ${SECTION_PADDING} flex justify-center px-4`}>
        <Button
          variant="ghost"
          size="sm"
          className="text-muted-foreground text-xs sm:text-sm"
          onClick={() => setOnboardingOpen(true)}
        >
          <Sparkles className="mr-2 h-3.5 w-3.5 sm:h-4 sm:w-4 shrink-0" />
          <span className="truncate">Frictionless onboarding preview</span>
        </Button>
      </div>

      <OnboardingPreviewModal open={onboardingOpen} onOpenChange={setOnboardingOpen} />
      <StickyCta />
    </div>
  );
}
