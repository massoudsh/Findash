'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ArrowRight, Sparkles } from 'lucide-react';
import { toast } from '@/components/ui/toast';

const SECTION_PADDING = 'py-16 sm:py-20';
const CONTAINER = 'container mx-auto px-4 sm:px-6 max-w-5xl';

export function StickyCta() {
  const [visible, setVisible] = useState(false);
  const [lastScroll, setLastScroll] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const y = window.scrollY;
      if (y > 400 && y > lastScroll) setVisible(true);
      else if (y < lastScroll || y < 200) setVisible(false);
      setLastScroll(y);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [lastScroll]);

  if (!visible) return null;

  return (
    <div
      className="fixed left-4 right-4 sm:left-1/2 sm:right-auto sm:max-w-sm z-50 sm:-translate-x-1/2 animate-in fade-in slide-in-from-bottom-4 duration-300"
      style={{ bottom: 'max(1rem, env(safe-area-inset-bottom, 1rem))' }}
    >
      <Link href="/trading" className="block w-full sm:w-auto">
        <Button
          size="default"
          className="w-full sm:w-auto h-10 sm:h-11 bg-amber-500 hover:bg-amber-600 text-amber-950 font-semibold shadow-lg border-0 text-sm sm:text-base"
        >
          Open مرکز فرماندهی
          <ArrowRight className="ml-2 h-4 w-4 shrink-0" />
        </Button>
      </Link>
    </div>
  );
}

export function FooterCtaBlock() {
  return (
    <section className={`${SECTION_PADDING} border-t border-border/60`}>
      <div className={`${CONTAINER} text-center`}>
        <h2 className="text-xl font-semibold text-foreground mb-2">Ready to take control?</h2>
        <p className="text-sm text-muted-foreground mb-6 max-w-lg mx-auto">
          One platform. Data, signals, risk, and execution. You decide.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-4">
          <Button asChild size="lg" className="bg-amber-500 hover:bg-amber-600 text-amber-950 font-semibold border-0">
            <Link href="/trading">
              Open مرکز فرماندهی
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
          <Button asChild size="lg" variant="outline">
            <Link href="/workflow">See how it works</Link>
          </Button>
        </div>
      </div>
    </section>
  );
}

export function EmailCaptureBlock() {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!email.trim()) return;
    setLoading(true);
    // Placeholder: in production, POST to your waitlist API
    setTimeout(() => {
      setLoading(false);
      setEmail('');
      toast({ title: "You're on the list", description: 'We’ll be in touch for beta access.', type: 'success' });
    }, 600);
  }

  return (
    <section className={`${SECTION_PADDING} bg-muted/30`}>
      <div className={`${CONTAINER} text-center px-4 sm:px-6`}>
        <h2 className="text-lg sm:text-xl font-semibold text-foreground mb-2">Join the beta waitlist</h2>
        <p className="text-sm text-muted-foreground mb-6 max-w-md mx-auto">
          Get early access and shape the product.
        </p>
        <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-3 max-w-sm mx-auto w-full">
          <Input
            type="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="flex-1"
            disabled={loading}
          />
          <Button type="submit" disabled={loading} className="bg-amber-500 hover:bg-amber-600 text-amber-950 border-0">
            {loading ? 'Joining…' : 'Join'}
          </Button>
        </form>
      </div>
    </section>
  );
}

export function OnboardingPreviewModal({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-amber-500" />
            Quick start
          </DialogTitle>
          <DialogDescription>
            Frictionless onboarding: connect data, set risk, then trade from the مرکز فرماندهی.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="rounded-lg border border-border/50 bg-muted/20 p-3 text-sm text-muted-foreground">
            <strong className="text-foreground">1.</strong> Add data sources (or use demo).
          </div>
          <div className="rounded-lg border border-border/50 bg-muted/20 p-3 text-sm text-muted-foreground">
            <strong className="text-foreground">2.</strong> Review signals and risk in Dashboard.
          </div>
          <div className="rounded-lg border border-border/50 bg-muted/20 p-3 text-sm text-muted-foreground">
            <strong className="text-foreground">3.</strong> Execute from مرکز فرماندهی when ready.
          </div>
        </div>
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
          <Button asChild className="bg-amber-500 hover:bg-amber-600 text-amber-950 border-0">
            <Link href="/trading" onClick={() => onOpenChange(false)}>
              Go to مرکز فرماندهی
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
