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
import { useBackendHealth } from '@/hooks/use-backend-health';
import { BackendOfflineBanner } from '@/components/ui/backend-offline-banner';

/** Best-practice strategy types (fin market aligned). */
const STRATEGY_TYPES = [
  { value: 'momentum', label: 'مومنتوم', desc: 'روند-محور؛ بهترین در بازارهای رونددار' },
  { value: 'mean_reversion', label: 'بازگشت به میانگین', desc: 'خرید/فروش در نقاط افراطی؛ بازارهای رنج' },
  { value: 'trend_following', label: 'دنباله‌روی روند', desc: 'سیستم‌های میانگین متحرک / شکست قیمتی' },
  { value: 'value', label: 'ارزشی', desc: 'سیگنال‌های بنیادی؛ افق زمانی بلندتر' },
  { value: 'arbitrage', label: 'آربیتراژ', desc: 'اختلاف قیمت بین بازارها؛ ریسک کم در هر معامله' },
  { value: 'scalping', label: 'اسکالپینگ', desc: 'کوتاه‌مدت؛ فرکانس بالا' },
] as const;

/** Agent sources that can feed the bot (M1–M11 alignment). */
const AGENT_SOURCES = [
  { value: 'm4', label: 'استراتژی (M4)', desc: 'سیگنال‌ها' },
  { value: 'm9', label: 'احساسات (M9)', desc: 'اخبار/شبکه‌های اجتماعی' },
  { value: 'm11', label: 'تحلیل (M11)', desc: 'بینش‌ها' },
  { value: 'm6', label: 'ریسک (M6)', desc: 'محدودیت‌های ریسک' },
] as const;

const STATUS_LABELS: Record<string, string> = {
  active: 'فعال',
  paused: 'متوقف‌شده',
  stopped: 'خاموش',
};

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
    name: 'ربات مومنتوم',
    strategy: 'momentum',
    status: 'active',
    executionMode: 'paper',
    symbols: ['NVDA', 'AAPL', 'MSFT'],
    agentSources: ['m4', 'm11'],
    risk: DEFAULT_RISK,
    performance: { total_trades: 45, win_rate: 0.67, total_pnl: 1250.5 },
    lastSignalAt: '۲ دقیقه پیش',
    created_at: new Date().toISOString(),
  },
  {
    id: '2',
    name: 'ربات بازگشت به میانگین',
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

const DEFAULT_BACKEND_URL = typeof process !== 'undefined' ? process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000' : 'http://localhost:8000';

export function TradingBotsContent() {
  const [bots, setBots] = useState<BotConfig[]>(MOCK_BOTS);
  const [botsLoading, setBotsLoading] = useState(true);
  const [botsError, setBotsError] = useState<string | null>(null);
  const { ok: backendOk, backendUrl, loading: backendHealthLoading, refetch: refetchBackendHealth } = useBackendHealth();

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
      setBotsError(e instanceof Error ? e.message : 'بارگذاری ربات‌ها ناموفق بود');
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
    if (!confirm('این ربات حذف شود؟ این عملیات قابل بازگشت نیست.')) return;
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
            ربات‌های معاملاتی
          </h1>
          <p className="text-muted-foreground mt-1">
            استراتژی‌های خودکار با کنترل ریسک؛ اجراشده روی پلتفرم از طریق مرکز فرماندهی
          </p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus className="h-4 w-4 mr-2" />
          ساخت ربات
        </Button>
      </div>

      {!backendHealthLoading && !API_BASE && (
        <BackendOfflineBanner
          backendUrl={DEFAULT_BACKEND_URL}
          message="اتصال بک‌اند: NEXT_PUBLIC_API_URL را در .env.local تنظیم کنید"
          fallbackLabel="(مثلاً http://localhost:8000). در حال استفاده از داده‌های آزمایشی محلی."
          onRetry={refetchBackendHealth}
        />
      )}
      {!backendHealthLoading && API_BASE && !backendOk && (
        <BackendOfflineBanner
          backendUrl={backendUrl}
          message="بک‌اند آفلاین است."
          fallbackLabel="نمایش داده‌های جایگزین."
          onRetry={refetchBackendHealth}
        />
      )}
      {botsError && (
        <Card className="border-destructive/30 bg-destructive/5">
          <CardContent className="py-3 px-4 flex items-center gap-3">
            <AlertTriangle className="h-5 w-5 text-destructive shrink-0" />
            <span className="text-sm">{botsError}. نمایش داده‌های جایگزین.</span>
            <Button variant="outline" size="sm" onClick={() => fetchBots()}>تلاش مجدد</Button>
          </CardContent>
        </Card>
      )}

      {/* Best-practice notice */}
      <Card className="border-amber-500/30 bg-amber-500/5">
        <CardContent className="py-3 px-4 flex items-start gap-3">
          <Shield className="h-5 w-5 text-amber-500 shrink-0 mt-0.5" />
          <div className="text-sm">
            <p className="font-medium">بهترین شیوه‌ها</p>
            <p className="text-muted-foreground">
              اندازه پوزیشن، حد ضرر، حد سود، حداکثر زیان روزانه و حداکثر افت سرمایه به‌ازای هر ربات اعمال می‌شود.
              ربات‌ها از سیگنال‌های عامل‌های استراتژی (M4)، احساسات (M9) و تحلیل (M11) استفاده می‌کنند. ابتدا در حالت
              <strong> آزمایشی</strong> اجرا کنید؛ زمانی که آماده بودید از تنظیمات ربات به حالت زنده تغییر دهید.
            </p>
          </div>
        </CardContent>
      </Card>

      {botsLoading ? (
        <div className="text-muted-foreground text-sm py-8 text-center">در حال بارگذاری ربات‌ها…</div>
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
                  {STATUS_LABELS[bot.status] ?? bot.status}
                </Badge>
              </div>
              <div className="flex flex-wrap gap-1 mt-1">
                <Badge variant="outline" className="text-xs">
                  {STRATEGY_TYPES.find((s) => s.value === bot.strategy)?.label ?? bot.strategy}
                </Badge>
                <Badge variant="outline" className="text-xs">
                  {bot.executionMode === 'paper' ? 'آزمایشی' : 'زنده'}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4 flex-1">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-muted-foreground">معاملات</span>
                  <p className="font-medium">{bot.performance.total_trades}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">نرخ برد</span>
                  <p className="font-medium">{(bot.performance.win_rate * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span className="text-muted-foreground">سود/زیان کل</span>
                  <p className={cn('font-medium', bot.performance.total_pnl >= 0 ? 'text-green-600' : 'text-red-600')}>
                    ${bot.performance.total_pnl.toFixed(2)}
                  </p>
                </div>
                <div>
                  <span className="text-muted-foreground">ریسک</span>
                  <p className="font-medium text-xs">
                    حد ضرر {bot.risk.stopLossPct}% · حد سود {bot.risk.takeProfitPct}% · حداکثر افت {bot.risk.maxDrawdownPct}%
                  </p>
                </div>
              </div>
              <div className="text-xs text-muted-foreground">
                <span>عامل‌ها: </span>
                {bot.agentSources.map((a) => AGENT_SOURCES.find((x) => x.value === a)?.label ?? a).join('، ')}
                {bot.symbols.length ? ` · ${bot.symbols.join('، ')}` : ''}
              </div>
              {bot.lastSignalAt && (
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Link2 className="h-3 w-3" />
                  آخرین سیگنال {bot.lastSignalAt} · اجرا روی پلتفرم
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
                  {bot.status === 'active' ? 'توقف موقت' : 'شروع'}
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
                  aria-label={`حذف ${bot.name}`}
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
            <DialogTitle>ساخت ربات معاملاتی</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <Label>نام ربات</Label>
              <Input
                placeholder="مثلاً مومنتوم فناوری"
                value={newBot.name}
                onChange={(e) => setNewBot((p) => ({ ...p, name: e.target.value }))}
                className="mt-1"
              />
            </div>
            <div>
              <Label>نوع استراتژی</Label>
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
              <Label>حالت اجرا</Label>
              <Select
                value={newBot.executionMode}
                onValueChange={(v) => setNewBot((p) => ({ ...p, executionMode: v as 'paper' | 'live' }))}
              >
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="paper">آزمایشی (ابتدا توصیه می‌شود)</SelectItem>
                  <SelectItem value="live">زنده</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>نمادها (جدا شده با کاما)</Label>
              <Input
                placeholder="NVDA, AAPL, SPY"
                value={newBot.symbols}
                onChange={(e) => setNewBot((p) => ({ ...p, symbols: e.target.value }))}
                className="mt-1"
              />
            </div>
            <div>
              <Label>ریسک: حد ضرر٪ · حد سود٪ · حداکثر زیان روزانه٪ · حداکثر افت سرمایه٪</Label>
              <div className="grid grid-cols-4 gap-2 mt-1">
                <Input
                  type="number"
                  step={0.5}
                  value={newBot.risk.stopLossPct}
                  onChange={(e) => setNewBot((p) => ({ ...p, risk: { ...p.risk, stopLossPct: Number(e.target.value) || 0 } }))}
                  placeholder="حد ضرر٪"
                />
                <Input
                  type="number"
                  step={0.5}
                  value={newBot.risk.takeProfitPct}
                  onChange={(e) => setNewBot((p) => ({ ...p, risk: { ...p.risk, takeProfitPct: Number(e.target.value) || 0 } }))}
                  placeholder="حد سود٪"
                />
                <Input
                  type="number"
                  step={0.5}
                  value={newBot.risk.maxDailyLossPct}
                  onChange={(e) => setNewBot((p) => ({ ...p, risk: { ...p.risk, maxDailyLossPct: Number(e.target.value) || 0 } }))}
                  placeholder="زیان روزانه٪"
                />
                <Input
                  type="number"
                  step={1}
                  value={newBot.risk.maxDrawdownPct}
                  onChange={(e) => setNewBot((p) => ({ ...p, risk: { ...p.risk, maxDrawdownPct: Number(e.target.value) || 0 } }))}
                  placeholder="افت سرمایه٪"
                />
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              ربات از سیگنال‌های عامل‌های استراتژی (M4)، تحلیل (M11) و به‌صورت اختیاری احساسات (M9) و ریسک (M6) استفاده می‌کند.
            </p>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>
              انصراف
            </Button>
            <Button onClick={createBot} disabled={!newBot.name.trim()}>
              ساخت
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
