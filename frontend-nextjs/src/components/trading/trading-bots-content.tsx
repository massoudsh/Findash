'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import {
  Bot,
  Plus,
  Play,
  Pause,
  Square,
  Shield,
  TrendingUp,
  BarChart3,
  Settings,
  Link2,
  AlertTriangle,
  Trash2,
} from 'lucide-react';
import { cn } from '@/lib/utils';

/** Best-practice strategy types (fin market aligned). */
const STRATEGY_TYPES = [
  { value: 'momentum', label: 'Momentum', desc: 'Trend-following; best in trending markets' },
  { value: 'mean_reversion', label: 'Mean Reversion', desc: 'Fade extremes; range-bound markets' },
  { value: 'trend_following', label: 'Trend Following', desc: 'Moving average / breakout systems' },
  { value: 'value', label: 'Value', desc: 'Fundamental signals; longer horizon' },
  { value: 'arbitrage', label: 'Arbitrage', desc: 'Price dislocations; low risk per trade' },
  { value: 'scalping', label: 'Scalping', desc: 'Short-term; high frequency' },
] as const;

/** Agent sources that can feed the bot (M1–M11 alignment). */
const AGENT_SOURCES = [
  { value: 'm4', label: 'Strategy (M4)', desc: 'Signals' },
  { value: 'm9', label: 'Sentiment (M9)', desc: 'News/social' },
  { value: 'm11', label: 'Analysis (M11)', desc: 'Insights' },
  { value: 'm6', label: 'Risk (M6)', desc: 'Risk limits' },
] as const;

interface BotConfig {
  id: string;
  name: string;
  strategy: string;
  status: 'active' | 'paused' | 'stopped';
  executionMode: 'paper' | 'live';
  symbols: string[];
  agentSources: string[];
  risk: {
    maxPositionPct: number;
    stopLossPct: number;
    takeProfitPct: number;
    maxDailyLossPct: number;
    maxDrawdownPct: number;
  };
  performance: { total_trades: number; win_rate: number; total_pnl: number };
  lastSignalAt?: string;
  created_at: string;
}

const DEFAULT_RISK = {
  maxPositionPct: 5,
  stopLossPct: 2,
  takeProfitPct: 4,
  maxDailyLossPct: 3,
  maxDrawdownPct: 10,
};

type NewBotForm = {
  name: string;
  strategy: string;
  executionMode: 'paper' | 'live';
  symbols: string;
  agentSources: string[];
  risk: typeof DEFAULT_RISK;
};

const MOCK_BOTS: BotConfig[] = [
  {
    id: '1',
    name: 'Momentum Bot',
    strategy: 'momentum',
    status: 'active',
    executionMode: 'paper',
    symbols: ['NVDA', 'AAPL', 'MSFT'],
    agentSources: ['m4', 'm11'],
    risk: DEFAULT_RISK,
    performance: { total_trades: 45, win_rate: 0.67, total_pnl: 1250.5 },
    lastSignalAt: '2m ago',
    created_at: new Date().toISOString(),
  },
  {
    id: '2',
    name: 'Mean Reversion Bot',
    strategy: 'mean_reversion',
    status: 'paused',
    executionMode: 'paper',
    symbols: ['SPY', 'QQQ'],
    agentSources: ['m4', 'm9', 'm6'],
    risk: { ...DEFAULT_RISK, maxPositionPct: 3, stopLossPct: 1.5 },
    performance: { total_trades: 32, win_rate: 0.59, total_pnl: 890.25 },
    created_at: new Date().toISOString(),
  },
];

const API_BASE = typeof process !== 'undefined' ? process.env.NEXT_PUBLIC_API_URL || '' : '';

function mapApiBotToConfig(b: {
  id: string;
  name: string;
  strategy: string;
  status: string;
  execution_mode?: string;
  symbols?: string[];
  agent_sources?: string[];
  risk?: Record<string, number>;
  performance?: { total_trades?: number; win_rate?: number; total_pnl?: number };
  last_signal_at?: string | null;
  created_at?: string;
}): BotConfig {
  const risk = b.risk || {};
  return {
    id: b.id,
    name: b.name,
    strategy: b.strategy,
    status: (b.status === 'active' || b.status === 'paused' || b.status === 'stopped' ? b.status : 'stopped') as BotConfig['status'],
    executionMode: (b.execution_mode === 'live' ? 'live' : 'paper') as 'paper' | 'live',
    symbols: Array.isArray(b.symbols) ? b.symbols : [],
    agentSources: Array.isArray(b.agent_sources) ? b.agent_sources : [],
    risk: {
      maxPositionPct: risk.max_position_pct ?? DEFAULT_RISK.maxPositionPct,
      stopLossPct: risk.stop_loss_pct ?? DEFAULT_RISK.stopLossPct,
      takeProfitPct: risk.take_profit_pct ?? DEFAULT_RISK.takeProfitPct,
      maxDailyLossPct: risk.max_daily_loss_pct ?? DEFAULT_RISK.maxDailyLossPct,
      maxDrawdownPct: risk.max_drawdown_pct ?? DEFAULT_RISK.maxDrawdownPct,
    },
    performance: {
      total_trades: b.performance?.total_trades ?? 0,
      win_rate: b.performance?.win_rate ?? 0,
      total_pnl: b.performance?.total_pnl ?? 0,
    },
    lastSignalAt: b.last_signal_at ?? undefined,
    created_at: b.created_at ?? new Date().toISOString(),
  };
}

export function TradingBotsContent() {
  const [bots, setBots] = useState<BotConfig[]>(MOCK_BOTS);
  const [botsLoading, setBotsLoading] = useState(true);
  const [botsError, setBotsError] = useState<string | null>(null);

  const fetchBots = useCallback(async () => {
    if (!API_BASE) {
      setBotsLoading(false);
      return;
    }
    setBotsLoading(true);
    setBotsError(null);
    try {
      const res = await fetch(`${API_BASE}/api/trading-bots/`, { credentials: 'include' });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      setBots(Array.isArray(data) ? data.map(mapApiBotToConfig) : MOCK_BOTS);
    } catch (e) {
      setBotsError(e instanceof Error ? e.message : 'Failed to load bots');
      setBots(MOCK_BOTS);
    } finally {
      setBotsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBots();
  }, [fetchBots]);
  const [createOpen, setCreateOpen] = useState(false);
  const [newBot, setNewBot] = useState<NewBotForm>({
    name: '',
    strategy: 'momentum',
    executionMode: 'paper',
    symbols: 'NVDA, AAPL',
    agentSources: ['m4', 'm11'],
    risk: { ...DEFAULT_RISK },
  });

  const createBot = async () => {
    if (!newBot.name.trim()) return;
    const payload = {
      name: newBot.name.trim(),
      strategy: newBot.strategy,
      executionMode: newBot.executionMode,
      symbols: newBot.symbols.split(/,\s*/).filter(Boolean),
      agentSources: newBot.agentSources,
      risk: {
        max_position_pct: newBot.risk.maxPositionPct,
        stop_loss_pct: newBot.risk.stopLossPct,
        take_profit_pct: newBot.risk.takeProfitPct,
        max_daily_loss_pct: newBot.risk.maxDailyLossPct,
        max_drawdown_pct: newBot.risk.maxDrawdownPct,
      },
    };
    if (API_BASE) {
      try {
        const res = await fetch(`${API_BASE}/api/trading-bots/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify(payload),
        });
        if (res.ok) {
          await fetchBots();
          setNewBot({ name: '', strategy: 'momentum', executionMode: 'paper', symbols: 'NVDA, AAPL', agentSources: ['m4', 'm11'], risk: { ...DEFAULT_RISK } });
          setCreateOpen(false);
          return;
        }
      } catch {
        // fallback to local state below
      }
    }
    setBots((prev) => [
      ...prev,
      {
        id: String(Date.now()),
        name: newBot.name.trim(),
        strategy: newBot.strategy,
        status: 'stopped',
        executionMode: newBot.executionMode,
        symbols: newBot.symbols.split(/,\s*/).filter(Boolean),
        agentSources: newBot.agentSources,
        risk: newBot.risk,
        performance: { total_trades: 0, win_rate: 0, total_pnl: 0 },
        created_at: new Date().toISOString(),
      },
    ]);
    setNewBot({ name: '', strategy: 'momentum', executionMode: 'paper', symbols: 'NVDA, AAPL', agentSources: ['m4', 'm11'], risk: { ...DEFAULT_RISK } });
    setCreateOpen(false);
  };

  const toggleStatus = async (id: string, current: string, forceNext?: 'active' | 'paused' | 'stopped') => {
    const next = forceNext ?? (current === 'active' ? 'paused' : 'active');
    if (API_BASE) {
      const path = next === 'active' ? 'start' : next === 'paused' ? 'pause' : 'stop';
      try {
        const res = await fetch(`${API_BASE}/api/trading-bots/${id}/${path}`, {
          method: 'POST',
          credentials: 'include',
        });
        if (res.ok) {
          await fetchBots();
          return;
        }
      } catch {
        // fallback to local state
      }
    }
    setBots((prev) => prev.map((b) => (b.id === id ? { ...b, status: next as BotConfig['status'] } : b)));
  };

  const deleteBot = async (id: string) => {
    if (!confirm('Delete this bot? This cannot be undone.')) return;
    if (API_BASE) {
      try {
        const res = await fetch(`${API_BASE}/api/trading-bots/${id}`, { method: 'DELETE', credentials: 'include' });
        if (res.ok) {
          await fetchBots();
          return;
        }
      } catch {
        // fallback to local state
      }
    }
    setBots((prev) => prev.filter((b) => b.id !== id));
  };

  return (
    <div className="space-y-6 p-6">
      <div className="flex justify-between items-start flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <Bot className="h-8 w-8" />
            Trading Bots
          </h1>
          <p className="text-muted-foreground mt-1">
            Automated strategies with risk controls; executed on the platform via Command Center
          </p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus className="h-4 w-4 mr-2" />
          Create Bot
        </Button>
      </div>

      {botsError && (
        <Card className="border-destructive/30 bg-destructive/5">
          <CardContent className="py-3 px-4 flex items-center gap-3">
            <AlertTriangle className="h-5 w-5 text-destructive shrink-0" />
            <span className="text-sm">{botsError}. Showing fallback data.</span>
            <Button variant="outline" size="sm" onClick={() => fetchBots()}>Retry</Button>
          </CardContent>
        </Card>
      )}

      {/* Best-practice notice */}
      <Card className="border-amber-500/30 bg-amber-500/5">
        <CardContent className="py-3 px-4 flex items-start gap-3">
          <Shield className="h-5 w-5 text-amber-500 shrink-0 mt-0.5" />
          <div className="text-sm">
            <p className="font-medium">Best practices</p>
            <p className="text-muted-foreground">
              Position sizing, stop-loss, take-profit, max daily loss and max drawdown are applied per bot.
              Bots consume signals from Strategy (M4), Sentiment (M9), and Analysis (M11) agents. Execute in
              <strong> Paper</strong> first; switch to Live from bot settings when ready.
            </p>
          </div>
        </CardContent>
      </Card>

      {botsLoading ? (
        <div className="text-muted-foreground text-sm py-8 text-center">Loading bots…</div>
      ) : (
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        {bots.map((bot) => (
          <Card key={bot.id} className="flex flex-col">
            <CardHeader className="pb-2">
              <div className="flex items-start justify-between gap-2">
                <CardTitle className="text-lg">{bot.name}</CardTitle>
                <Badge
                  variant={bot.status === 'active' ? 'default' : bot.status === 'paused' ? 'secondary' : 'outline'}
                  className="shrink-0"
                >
                  {bot.status}
                </Badge>
              </div>
              <div className="flex flex-wrap gap-1 mt-1">
                <Badge variant="outline" className="text-xs">
                  {STRATEGY_TYPES.find((s) => s.value === bot.strategy)?.label ?? bot.strategy}
                </Badge>
                <Badge variant="outline" className="text-xs">
                  {bot.executionMode === 'paper' ? 'Paper' : 'Live'}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4 flex-1">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-muted-foreground">Trades</span>
                  <p className="font-medium">{bot.performance.total_trades}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Win rate</span>
                  <p className="font-medium">{(bot.performance.win_rate * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Total P&L</span>
                  <p className={cn('font-medium', bot.performance.total_pnl >= 0 ? 'text-green-600' : 'text-red-600')}>
                    ${bot.performance.total_pnl.toFixed(2)}
                  </p>
                </div>
                <div>
                  <span className="text-muted-foreground">Risk</span>
                  <p className="font-medium text-xs">
                    SL {bot.risk.stopLossPct}% · TP {bot.risk.takeProfitPct}% · Max DD {bot.risk.maxDrawdownPct}%
                  </p>
                </div>
              </div>
              <div className="text-xs text-muted-foreground">
                <span>Agents: </span>
                {bot.agentSources.map((a) => AGENT_SOURCES.find((x) => x.value === a)?.label ?? a).join(', ')}
                {bot.symbols.length ? ` · ${bot.symbols.join(', ')}` : ''}
              </div>
              {bot.lastSignalAt && (
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Link2 className="h-3 w-3" />
                  Last signal {bot.lastSignalAt} · runs on platform
                </p>
              )}
              <div className="flex gap-2 pt-2">
                <Button
                  variant={bot.status === 'active' ? 'secondary' : 'default'}
                  size="sm"
                  className="flex-1"
                  onClick={() => toggleStatus(bot.id, bot.status)}
                  disabled={bot.status === 'stopped'}
                >
                  {bot.status === 'active' ? <Pause className="h-3 w-3 mr-1" /> : <Play className="h-3 w-3 mr-1" />}
                  {bot.status === 'active' ? 'Pause' : 'Start'}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => toggleStatus(bot.id, bot.status, 'stopped')}
                  disabled={bot.status === 'stopped'}
                >
                  <Square className="h-3 w-3" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground hover:text-destructive"
                  onClick={() => deleteBot(bot.id)}
                  aria-label={`Delete ${bot.name}`}
                >
                  <Trash2 className="h-3 w-3" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
      )}

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Create Trading Bot</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <Label>Bot name</Label>
              <Input
                placeholder="e.g. Tech Momentum"
                value={newBot.name}
                onChange={(e) => setNewBot((p) => ({ ...p, name: e.target.value }))}
                className="mt-1"
              />
            </div>
            <div>
              <Label>Strategy type</Label>
              <Select
                value={newBot.strategy}
                onValueChange={(v) => setNewBot((p) => ({ ...p, strategy: v as typeof p.strategy }))}
              >
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {STRATEGY_TYPES.map((s) => (
                    <SelectItem key={s.value} value={s.value}>
                      {s.label} — {s.desc}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Execution mode</Label>
              <Select
                value={newBot.executionMode}
                onValueChange={(v) => setNewBot((p) => ({ ...p, executionMode: v as 'paper' | 'live' }))}
              >
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="paper">Paper (recommended first)</SelectItem>
                  <SelectItem value="live">Live</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Symbols (comma-separated)</Label>
              <Input
                placeholder="NVDA, AAPL, SPY"
                value={newBot.symbols}
                onChange={(e) => setNewBot((p) => ({ ...p, symbols: e.target.value }))}
                className="mt-1"
              />
            </div>
            <div>
              <Label>Risk: Stop-loss % · Take-profit % · Max daily loss % · Max drawdown %</Label>
              <div className="grid grid-cols-4 gap-2 mt-1">
                <Input
                  type="number"
                  step={0.5}
                  value={newBot.risk.stopLossPct}
                  onChange={(e) => setNewBot((p) => ({ ...p, risk: { ...p.risk, stopLossPct: Number(e.target.value) || 0 } }))}
                  placeholder="SL %"
                />
                <Input
                  type="number"
                  step={0.5}
                  value={newBot.risk.takeProfitPct}
                  onChange={(e) => setNewBot((p) => ({ ...p, risk: { ...p.risk, takeProfitPct: Number(e.target.value) || 0 } }))}
                  placeholder="TP %"
                />
                <Input
                  type="number"
                  step={0.5}
                  value={newBot.risk.maxDailyLossPct}
                  onChange={(e) => setNewBot((p) => ({ ...p, risk: { ...p.risk, maxDailyLossPct: Number(e.target.value) || 0 } }))}
                  placeholder="Daily %"
                />
                <Input
                  type="number"
                  step={1}
                  value={newBot.risk.maxDrawdownPct}
                  onChange={(e) => setNewBot((p) => ({ ...p, risk: { ...p.risk, maxDrawdownPct: Number(e.target.value) || 0 } }))}
                  placeholder="DD %"
                />
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Bot will use signals from Strategy (M4), Analysis (M11), and optional Sentiment (M9) and Risk (M6) agents.
            </p>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>
              Cancel
            </Button>
            <Button onClick={createBot} disabled={!newBot.name.trim()}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
