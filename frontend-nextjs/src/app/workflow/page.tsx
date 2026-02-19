'use client';

import { useEffect, useRef } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ArrowRight, Sparkles, LayoutDashboard, Target, Database, Cloud, Brain, User, Zap, BarChart3, ChevronRight } from 'lucide-react';
import { FalloutCharacter } from '@/components/ui/fallout-character';
import { WorkflowFlowChart } from '@/components/workflow/workflow-flow-chart';

declare global {
  interface Window {
    mermaid?: {
      run: (config?: { nodes: HTMLElement[] }) => Promise<void>;
      initialize: (config: Record<string, unknown>) => void;
    };
  }
}

const MERMAID_SCRIPT = 'https://cdn.jsdelivr.net/npm/mermaid@9/dist/mermaid.min.js';

const diagram1 = `flowchart LR
    subgraph SOURCES["📡 Sources"]
        MKT[Markets]
        NEWS[News & Social]
        ALT[Alternative Data]
    end
    subgraph INGEST["🔄 Ingest & Store"]
        M1[Nexus M1]
        M2[Vault M2]
        M3[Pulse M3]
        M9[Echo M9]
    end
    subgraph ANALYZE["🧠 Analyze"]
        M5[Neuron M5]
        M7[Oracle M7]
        M4[Atlas M4]
        M6[Guardian M6]
    end
    subgraph DECIDE["👤 Decide"]
        USER[Trader]
    end
    subgraph EXECUTE["📤 Execute & Report"]
        M8[Shadow M8]
        M10[Chronicle M10]
        M11[Lens M11]
    end
    MKT & NEWS & ALT --> M1
    M1 --> M2
    M2 --> M3
    M1 --> M9
    M3 --> M5 & M7
    M9 --> M4
    M5 & M7 --> M4
    M4 --> M6
    M6 --> USER
    USER --> M8 & M10
    M8 --> M11
    M10 --> M11
    M11 --> REPORTS[Reports & Dashboards]`;

const diagram2 = `flowchart TB
    subgraph Phase1["Phase 1: Market updates"]
        A1["📡 Nexus M1: Ingest prices, news, alt data"]
        A2["🗄️ Vault M2: Store & validate"]
        A3["⚡ Pulse M3: Stream to platform"]
        A4["💬 Echo M9: Sentiment scores"]
        A1 --> A2 --> A3
        A1 --> A4
    end
    subgraph Phase2["Phase 2: Analytics & signals"]
        B1["🧠 Neuron M5: ML predictions"]
        B2["🔮 Oracle M7: Price forecasts"]
        B3["🎯 Atlas M4: Trading signals"]
        B4["🛡️ Guardian M6: VaR, sizing, limits"]
        B1 & B2 --> B3
        B3 --> B4
    end
    subgraph Phase3["Phase 3: Your decision"]
        C1[Dashboard & Command Center]
        C2[Review signals, risk, portfolio]
        C3[Approve / Reject / Modify]
        C1 --> C2 --> C3
    end
    subgraph Phase4["Phase 4: Execution & validation"]
        D1["📋 Shadow M8: Paper or live execution"]
        D2["📜 Chronicle M10: Backtest if needed"]
        D3["📊 Lens M11: Build views & reports"]
        D1 --> D3
        D2 --> D3
    end
    Phase1 --> Phase2 --> Phase3 --> Phase4
    Phase4 -.->|Feedback| Phase1`;

const diagram3 = `sequenceDiagram
    participant User as 👤 Trader
    participant UI as Dashboard / Command Center
    participant M4 as Atlas Strategy
    participant M6 as Guardian Risk
    participant M8 as Shadow Execution
    participant M11 as Lens Reports
    M4->>UI: Signals & strategy suggestions
    M6->>UI: Risk view, position sizing, limits
    UI->>User: Present options & risk
    User->>UI: Choose: approve / reject / modify
    UI->>M8: Send order
    M8->>UI: Fills & position update
    UI->>User: Confirmation
    User->>UI: Request report or dashboard
    UI->>M11: Build visualization / report
    M11->>UI: Charts, tables, insights
    UI->>User: Report & visualization`;

export default function WorkflowPage() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (typeof window === 'undefined' || !containerRef.current) return;
    const loadMermaid = () => {
      if (!window.mermaid) {
        const script = document.createElement('script');
        script.src = MERMAID_SCRIPT;
        script.async = true;
        script.onload = () => runMermaid();
        document.head.appendChild(script);
      } else {
        runMermaid();
      }
    };
    const runMermaid = () => {
      try {
        const isDark = document.documentElement.classList.contains('dark');
        window.mermaid?.initialize({
          startOnLoad: false,
          theme: 'base',
          themeVariables: isDark
            ? {
                darkMode: true,
                background: '#1c1917',
                primaryColor: '#fef3c7',
                primaryTextColor: '#fef9c3',
                primaryBorderColor: '#f59e0b',
                secondaryColor: '#e0e7ff',
                secondaryTextColor: '#e0e7ff',
                tertiaryColor: '#d1fae5',
                lineColor: '#94a3b8',
                textColor: '#f1f5f9',
                fontFamily: 'ui-sans-serif, system-ui, sans-serif',
                fontSize: '15px',
              }
            : {
                darkMode: false,
                background: '#fafaf9',
                primaryColor: '#fef3c7',
                primaryTextColor: '#1c1917',
                primaryBorderColor: '#f59e0b',
                secondaryColor: '#e0e7ff',
                secondaryTextColor: '#312e81',
                tertiaryColor: '#d1fae5',
                lineColor: '#475569',
                textColor: '#1c1917',
                fontFamily: 'ui-sans-serif, system-ui, sans-serif',
                fontSize: '15px',
              },
          flowchart: { useMaxWidth: true, padding: 16 },
          sequence: { useMaxWidth: true, diagramMarginX: 20, diagramMarginY: 20 },
        });
        window.mermaid?.run?.({ nodes: Array.from(containerRef.current!.querySelectorAll('.mermaid')) as HTMLElement[] });
      } catch (e) {
        console.warn('Mermaid render:', e);
      }
    };
    loadMermaid();
  }, []);

  return (
    <div className="container mx-auto px-6 py-8 space-y-8 max-w-5xl">
      {/* Fantasy story: Octopus, his legs, and resilience */}
      <div className="rounded-2xl border-2 border-amber-400/60 bg-gradient-to-br from-amber-50/80 to-yellow-100/80 dark:from-amber-950/30 dark:to-yellow-900/10 p-6 space-y-5">
        <div className="inline-flex items-center gap-2 rounded-full bg-amber-400/20 px-4 py-1.5 text-sm font-medium text-amber-800 dark:text-amber-200">
          <FalloutCharacter pose="storm" size={24} />
          <Sparkles className="h-4 w-4" />
          The Octopus and the storm
        </div>
        <div className="grid gap-4 sm:grid-cols-1 md:grid-cols-3 text-left">
          <div className="rounded-xl border border-amber-300/50 bg-white/60 dark:bg-black/20 p-4 flex gap-3">
            <FalloutCharacter pose="arms" size={36} className="shrink-0 mt-0.5" />
            <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-300 mb-2">His many arms</p>
            <p className="text-sm text-foreground">In the deep, noisy waters of the market, our Octopus doesn’t drown. Each arm has a job: one gathers data, another reads signals, others guard risk or run the numbers. Together they hold everything you need in one place.</p>
            </div>
          </div>
          <div className="rounded-xl border border-amber-300/50 bg-white/60 dark:bg-black/20 p-4 flex gap-3">
            <FalloutCharacter pose="resilience" size={36} className="shrink-0 mt-0.5" />
            <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-300 mb-2">Resilience</p>
            <p className="text-sm text-foreground">When the waters get rough — volatility, news, chaos — the Octopus doesn’t let go. He adapts, holds the line, and keeps your view clear. No matter how the market moves, he stays steady so you can decide without the panic.</p>
            </div>
          </div>
          <div className="rounded-xl border border-amber-300/50 bg-white/60 dark:bg-black/20 p-4 flex gap-3">
            <FalloutCharacter pose="helm" size={36} className="shrink-0 mt-0.5" />
            <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-300 mb-2">You at the helm</p>
            <p className="text-sm text-foreground">The Octopus prepares; you command. In the Command Center you see what his arms have gathered — signals, risk, reports — and you make the call. He works so you stay in control. That’s the pact.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Hero CTA - yellow accent */}
      <div className="relative rounded-2xl border-2 border-amber-400/80 bg-gradient-to-br from-amber-50 to-yellow-100 dark:from-amber-950/40 dark:to-yellow-900/20 p-8 text-center shadow-lg">
        <div className="inline-flex items-center gap-2 rounded-full bg-amber-400/20 px-4 py-1.5 text-sm font-medium text-amber-800 dark:text-amber-200 mb-4">
          <FalloutCharacter pose="how" size={24} />
          <Sparkles className="h-4 w-4" />
          How Octopus works
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          You decide. Agents assist.
        </h1>
        <p className="mt-3 text-muted-foreground max-w-xl mx-auto">
          From live market data to signals and risk — our 11 agents prepare everything. You review in the Command Center and take the final call.
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-4">
          <Button
            asChild
            className="bg-amber-500 hover:bg-amber-600 text-amber-950 font-semibold shadow-md border-0"
          >
            <Link href="/trading">
              Open Command Center
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
          <Button asChild variant="outline" className="border-amber-500/60 text-amber-700 dark:text-amber-300 hover:bg-amber-500/10">
            <Link href="/dashboard">
              <LayoutDashboard className="mr-2 h-4 w-4" />
              Go to Dashboard
            </Link>
          </Button>
        </div>
      </div>

      {/* One-line overview - user friendly */}
      <Card className="border-amber-500/20 bg-amber-50/50 dark:bg-amber-950/20">
        <CardContent className="pt-6">
          <p className="text-center text-sm font-medium text-foreground">
            <span className="text-amber-600 dark:text-amber-400">Data in</span>
            {' → '}
            <span className="text-amber-600 dark:text-amber-400">Agents analyze</span>
            {' → '}
            <span className="text-amber-600 dark:text-amber-400">You decide</span>
            {' → '}
            <span className="text-amber-600 dark:text-amber-400">Execution & reports</span>
          </p>
        </CardContent>
      </Card>

      {/* Infographic: End-to-end pipeline (Google AI style) */}
      <div className="rounded-2xl border border-border/50 bg-card/50 backdrop-blur-sm shadow-lg shadow-black/5 dark:shadow-black/20 p-6">
        <h3 className="text-lg font-semibold tracking-tight text-foreground mb-1">End-to-end pipeline</h3>
        <p className="text-sm text-muted-foreground mb-6">From sources to reports in one flow</p>
        <div className="flex flex-col sm:flex-row sm:items-stretch gap-4 sm:gap-2 overflow-x-auto pb-2">
          {[
            { icon: Database, label: 'Sources', desc: 'Markets, news, alt data', color: 'bg-blue-500/10 text-blue-600 dark:text-blue-400' },
            { icon: Cloud, label: 'Ingest', desc: 'Store & stream', color: 'bg-violet-500/10 text-violet-600 dark:text-violet-400' },
            { icon: Brain, label: 'Analyze', desc: 'Signals & risk', color: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400' },
            { icon: User, label: 'You', desc: 'Decide', color: 'bg-amber-500/10 text-amber-600 dark:text-amber-400' },
            { icon: Zap, label: 'Execute', desc: 'Orders & backtest', color: 'bg-orange-500/10 text-orange-600 dark:text-orange-400' },
            { icon: BarChart3, label: 'Reports', desc: 'Dashboards & charts', color: 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400' },
          ].map(({ icon: Icon, label, desc, color }, i) => (
            <div key={label} className="flex sm:flex-1 sm:min-w-0 items-center gap-2">
              {i > 0 && <ChevronRight className="h-5 w-5 shrink-0 text-muted-foreground/40 hidden sm:block" />}
              <div className={`flex flex-col items-center justify-center gap-2 rounded-xl p-4 flex-1 min-w-[120px] ${color} border border-border/30 hover:border-border/60 transition-colors`}>
                <div className={`rounded-full p-2.5 ${color}`}>
                  <Icon className="h-5 w-5" />
                </div>
                <p className="font-semibold text-sm text-foreground">{label}</p>
                <p className="text-xs text-muted-foreground text-center">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Visual workflow — Figma-style (React Flow), fits container */}
      <Card className="mt-6 border-l-4 border-l-amber-500">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Target className="h-5 w-5 text-amber-500" />
            Visual workflow
          </CardTitle>
          <p className="text-sm text-muted-foreground">Pan, zoom, and fit to view. Sources → Ingest → Analyze → You → Execute.</p>
        </CardHeader>
        <CardContent>
          <WorkflowFlowChart />
        </CardContent>
      </Card>

      {/* Diagrams - Mermaid (detailed, code-based) */}
      <div ref={containerRef}>
        <Card className="mt-6 border-l-4 border-l-amber-500 overflow-hidden">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-lg">
              Detailed pipeline (Mermaid)
            </CardTitle>
            <p className="text-sm text-muted-foreground">Full agent flow — Sources → Ingest → Analyze → You → Execute</p>
          </CardHeader>
          <CardContent className="p-0">
            <div
              className="mermaid-diagram-container min-h-[320px] overflow-auto rounded-b-lg border-t border-border bg-stone-100/80 dark:bg-stone-900/60 p-8 flex items-center justify-center ring-1 ring-inset ring-border/30"
              aria-label="Detailed pipeline flowchart"
            >
              <pre id="mermaid-pipeline" className="mermaid text-sm m-0 flex items-center justify-center [&_svg]:max-w-full [&_svg]:h-auto">
                {diagram1}
              </pre>
            </div>
          </CardContent>
        </Card>

        <Card className="mt-6 border-l-4 border-l-amber-500 overflow-hidden">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Decision flow in 4 phases</CardTitle>
            <p className="text-sm text-muted-foreground">Market updates → Analytics → Your decision → Execution & reporting</p>
          </CardHeader>
          <CardContent className="p-0">
            <div
              className="mermaid-diagram-container min-h-[380px] overflow-auto rounded-b-lg border-t border-border bg-stone-100/80 dark:bg-stone-900/60 p-8 flex items-center justify-center ring-1 ring-inset ring-border/30"
              aria-label="Decision flow in 4 phases"
            >
              <pre id="mermaid-phases" className="mermaid text-sm m-0 flex items-center justify-center [&_svg]:max-w-full [&_svg]:h-auto">
                {diagram2}
              </pre>
            </div>
          </CardContent>
        </Card>

        <Card className="mt-6 border-l-4 border-l-amber-500 overflow-hidden">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">You and the agents (sequence)</CardTitle>
            <p className="text-sm text-muted-foreground">How the UI and agents interact when you trade or request a report</p>
          </CardHeader>
          <CardContent className="p-0">
            <div
              className="mermaid-diagram-container min-h-[340px] overflow-auto rounded-b-lg border-t border-border bg-stone-100/80 dark:bg-stone-900/60 p-8 flex items-center justify-center ring-1 ring-inset ring-border/30"
              aria-label="You and the agents sequence diagram"
            >
              <pre id="mermaid-sequence" className="mermaid text-sm m-0 flex items-center justify-center [&_svg]:max-w-full [&_svg]:h-auto">
                {diagram3}
              </pre>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Step-by-step - cleaner table with yellow row for "You" */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">12 steps: market update → report</CardTitle>
          <p className="text-sm text-muted-foreground">Who does what, in order</p>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto rounded-lg border border-border">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="text-left p-3 font-medium">Step</th>
                  <th className="text-left p-3 font-medium">Phase</th>
                  <th className="text-left p-3 font-medium">Who</th>
                  <th className="text-left p-3 font-medium">Action</th>
                  <th className="text-left p-3 font-medium">Output</th>
                </tr>
              </thead>
              <tbody>
                {[
                  [1, 'Market updates', 'Nexus (M1)', 'Ingest prices, news, social', 'Normalized data'],
                  [2, 'Market updates', 'Vault (M2)', 'Store & validate', 'Historical + real-time datasets'],
                  [3, 'Market updates', 'Pulse (M3)', 'Stream live data', 'WebSocket feeds, live P&L'],
                  [4, 'Market updates', 'Echo (M9)', 'Sentiment scores', 'Signals per asset/theme'],
                  [5, 'Analytics', 'Neuron (M5)', 'ML models', 'Predictions, regime labels'],
                  [6, 'Analytics', 'Oracle (M7)', 'Price forecasting', 'Targets, scenarios'],
                  [7, 'Signals', 'Atlas (M4)', 'Fuse signals, ideas', 'Trading signals, suggestions'],
                  [8, 'Risk', 'Guardian (M6)', 'VaR, limits, sizing', 'Approved size, risk view'],
                  [9, 'Decision', '👤 You', 'Approve, reject, or adjust in Command Center', 'Your order or no-trade'],
                  [10, 'Execution', 'Shadow (M8)', 'Paper or live execution', 'Fills, position updates'],
                  [11, 'Validation', 'Chronicle (M10)', 'Backtest if needed', 'Backtest report'],
                  [12, 'Reporting', 'Lens (M11)', 'Charts, dashboards, AI reports', 'Visualizations & reports'],
                ].map(([step, phase, who, action, output]) => (
                  <tr
                    key={String(step)}
                    className={`border-b border-border/50 ${who === '👤 You' ? 'bg-amber-100/60 dark:bg-amber-900/20' : ''}`}
                  >
                    <td className="p-3 font-medium">{step}</td>
                    <td className="p-3">{phase}</td>
                    <td className="p-3 font-medium">{who}</td>
                    <td className="p-3 text-muted-foreground">{action}</td>
                    <td className="p-3 text-muted-foreground">{output}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Where to find each agent - compact cards */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Where to find each agent</CardTitle>
          <p className="text-sm text-muted-foreground">Quick reference: which page shows which agent</p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {[
              ['Nexus (M1)', 'Data Explorer, Command Center', 'Live feeds, export'],
              ['Vault (M2)', 'Data Explorer', 'Historical data'],
              ['Pulse (M3)', 'Dashboard, live charts', 'Real-time prices'],
              ['Atlas (M4)', 'Command Center → Strategies, Bots', 'Signals, ideas'],
              ['Neuron (M5)', 'AI Models', 'Predictions'],
              ['Guardian (M6)', 'Command Center → Risk', 'VaR, limits'],
              ['Oracle (M7)', 'Options', 'Price forecasts'],
              ['Shadow (M8)', 'Paper trading, Portfolio', 'Sim execution'],
              ['Echo (M9)', 'Command Center, Social', 'Sentiment'],
              ['Chronicle (M10)', 'Command Center → Backtesting', 'Backtest results'],
              ['Lens (M11)', 'Reports, Visualization, Dashboard', 'Charts, insights'],
            ].map(([agent, where, what]) => (
              <div
                key={agent}
                className="flex items-start gap-3 rounded-lg border border-border bg-card p-3 hover:border-amber-500/30 transition-colors"
              >
                <span className="shrink-0 rounded bg-amber-500/15 px-2 py-0.5 text-xs font-medium text-amber-700 dark:text-amber-300">
                  {agent}
                </span>
                <div className="min-w-0">
                  <p className="text-sm font-medium">{where}</p>
                  <p className="text-xs text-muted-foreground">{what}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Bottom CTA + summary */}
      <Card className="border-2 border-amber-500/30 bg-amber-50/30 dark:bg-amber-950/20">
        <CardHeader>
          <CardTitle className="text-lg">In a nutshell</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <ul className="text-sm text-muted-foreground space-y-2 list-disc list-inside">
            <li><strong className="text-foreground">Agents</strong> handle data, analytics, signals, risk, execution, and reporting.</li>
            <li><strong className="text-foreground">You</strong> decide in the Command Center and Dashboard — approve, reject, or adjust.</li>
            <li><strong className="text-foreground">Flow:</strong> Market updates → enrichment → signals & risk → your decision → execution → reports.</li>
          </ul>
          <div className="pt-2">
            <Button asChild className="bg-amber-500 hover:bg-amber-600 text-amber-950 font-semibold">
              <Link href="/trading">
                Open Command Center
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
