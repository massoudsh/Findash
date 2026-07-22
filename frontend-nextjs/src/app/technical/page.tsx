'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { TradingViewChart } from '@/components/charts/tradingview-chart';
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  Activity,
  Target,
  Zap,
  Eye,
  Settings,
  LineChart,
  Gauge,
  AreaChart,
  BarChart,
  Calendar,
  Clock,
  Layers,
  Maximize2,
  Volume2,
  RefreshCw,
  Download,
  Share2,
  Bookmark,
  AlertTriangle,
  Minus,
  Star,
} from 'lucide-react';

const SYMBOL_TO_API = (s: string) => s.replace('/', '-');
const API_SYMBOLS = 'BTC-USD,ETH-USD,AAPL,TSLA,NVDA,MSFT,GOOGL';
const WATCHLIST_STORAGE_KEY = 'technical-watchlist';

const DEFAULT_WATCHLIST = ['BTC/USD', 'ETH/USD', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'];

function loadWatchlist(): string[] {
  if (typeof window === 'undefined') return DEFAULT_WATCHLIST;
  try {
    const raw = localStorage.getItem(WATCHLIST_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as string[];
      return Array.isArray(parsed) && parsed.length > 0 ? parsed : DEFAULT_WATCHLIST;
    }
  } catch (_) {}
  return DEFAULT_WATCHLIST;
}

function saveWatchlist(symbols: string[]) {
  try {
    localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(symbols));
  } catch (_) {}
}

const ECONOMIC_EVENTS_MOCK = [
  { title: 'بیانیه FOMC', date: 'این هفته', impact: 'بالا' },
  { title: 'اشتغال غیرکشاورزی', date: 'جمعه آینده', impact: 'بالا' },
  { title: 'انتشار CPI', date: 'ماه آینده', impact: 'بالا' },
  { title: 'سخنرانی رئیس فدرال رزرو', date: 'دو هفته دیگر', impact: 'متوسط' },
  { title: 'فروش خرده‌فروشی', date: 'هفته آینده', impact: 'متوسط' },
  { title: 'درخواست‌های اولیه بیمه بیکاری', date: 'پنجشنبه', impact: 'پایین' },
];

interface EconomicEvent {
  title: string;
  date: string;
  impact: string;
}

export default function TechnicalPage() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USD');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D');
  const [chartType, setChartType] = useState('candlestick');
  const [showVolume, setShowVolume] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [chartRefreshKey, setChartRefreshKey] = useState(0);
  const [realMarketData, setRealMarketData] = useState<Record<string, { price: number; open?: number; high?: number; low?: number; volume?: number; change?: number; change_percent?: number }>>({});
  const [activePanel, setActivePanel] = useState<'screener' | 'watchlist' | 'calendar' | 'chartSettings' | 'marketOverview' | null>(null);
  const [watchlist, setWatchlist] = useState<string[]>(DEFAULT_WATCHLIST);
  const [watchlistLoaded, setWatchlistLoaded] = useState(false);
  const [economicEvents, setEconomicEvents] = useState<EconomicEvent[]>(ECONOMIC_EVENTS_MOCK);
  const [calendarLoading, setCalendarLoading] = useState(false);

  useEffect(() => {
    setWatchlist(loadWatchlist());
    setWatchlistLoaded(true);
  }, []);

  const persistWatchlist = (symbols: string[]) => {
    setWatchlist(symbols);
    saveWatchlist(symbols);
  };

  const addToWatchlist = (symbol: string) => {
    if (watchlist.includes(symbol)) return;
    persistWatchlist([...watchlist, symbol]);
  };

  const removeFromWatchlist = (symbol: string) => {
    persistWatchlist(watchlist.filter((s) => s !== symbol));
  };

  const fetchRealMarketData = useCallback(async () => {
    try {
      const res = await fetch(`/api/real-market-data?symbols=${API_SYMBOLS}`);
      const json = await res.json();
      if (json?.data) setRealMarketData(json.data);
    } catch (e) {
      console.error('Failed to fetch real market data', e);
    }
  }, []);

  useEffect(() => {
    fetchRealMarketData();
  }, [fetchRealMarketData]);

  const fetchEconomicCalendar = useCallback(async () => {
    setCalendarLoading(true);
    try {
      const res = await fetch('/api/economic-calendar?days_ahead=14');
      const json = await res.json();
      if (res.ok && Array.isArray(json?.data) && json.data.length > 0) {
        setEconomicEvents(
          json.data.map((ev: { event: string; date?: string; time?: string; impact?: string }) => ({
            title: ev.event,
            date: [ev.date, ev.time].filter(Boolean).join(' ') || '—',
            impact: (ev.impact ?? 'medium').charAt(0).toUpperCase() + (ev.impact ?? 'medium').slice(1),
          }))
        );
      }
    } catch {
      setEconomicEvents(ECONOMIC_EVENTS_MOCK);
    } finally {
      setCalendarLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activePanel === 'calendar') fetchEconomicCalendar();
  }, [activePanel, fetchEconomicCalendar]);

  const currentSymbolKey = SYMBOL_TO_API(selectedSymbol);
  const currentMarket = realMarketData[currentSymbolKey];

  // Mock price data for chart simulation
  const generateMockData = () => {
    const data = [];
    let price = 42350;
    for (let i = 0; i < 100; i++) {
      const change = (Math.random() - 0.5) * 1000;
      price += change;
      data.push({
        time: new Date(Date.now() - (100 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        open: price - Math.random() * 500,
        high: price + Math.random() * 800,
        low: price - Math.random() * 800,
        close: price,
        volume: Math.floor(Math.random() * 1000000) + 500000
      });
    }
    return data;
  };

  const chartData = generateMockData();

  const technicalIndicators = [
    {
      name: 'RSI (14)',
      value: 68.5,
      signal: 'اشباع خرید',
      color: 'orange',
      description: 'شاخص قدرت نسبی',
      recommendation: 'سیگنال فروش'
    },
    {
      name: 'MACD (12,26,9)',
      value: 0.45,
      signal: 'تقاطع صعودی',
      color: 'green',
      description: 'همگرایی/واگرایی میانگین متحرک',
      recommendation: 'سیگنال خرید'
    },
    {
      name: 'Stochastic (14,3,3)',
      value: 75.2,
      signal: 'اشباع خرید',
      color: 'red',
      description: 'نوسان‌ساز استوکاستیک',
      recommendation: 'سیگنال فروش'
    },
    {
      name: 'Williams %R (14)',
      value: -25.8,
      signal: 'صعودی',
      color: 'green',
      description: 'محدوده درصدی ویلیامز',
      recommendation: 'سیگنال خرید'
    },
    {
      name: 'CCI (20)',
      value: 145.6,
      signal: 'اشباع خرید',
      color: 'orange',
      description: 'شاخص کانال کالا',
      recommendation: 'خنثی'
    },
    {
      name: 'ADX (14)',
      value: 32.1,
      signal: 'روند قوی',
      color: 'blue',
      description: 'شاخص جهت‌دار میانگین',
      recommendation: 'پیروی از روند'
    },
    {
      name: 'Bollinger Bands',
      value: 0.82,
      signal: 'باند بالایی',
      color: 'purple',
      description: 'شاخص نوسان',
      recommendation: 'سیگنال فروش'
    },
    {
      name: 'ATR (14)',
      value: 1250.5,
      signal: 'نوسان بالا',
      color: 'yellow',
      description: 'میانگین محدوده واقعی',
      recommendation: 'احتیاط'
    }
  ];

  const movingAverages = [
    { period: 'SMA 20', value: '$42,150', position: 'Above', signal: 'صعودی', distance: '۱.۲%+' },
    { period: 'SMA 50', value: '$41,850', position: 'Above', signal: 'صعودی', distance: '۲.۸%+' },
    { period: 'SMA 200', value: '$39,200', position: 'Above', signal: 'صعودی', distance: '۸.۰%+' },
    { period: 'EMA 12', value: '$42,280', position: 'Above', signal: 'صعودی', distance: '۰.۹%+' },
    { period: 'EMA 26', value: '$41,950', position: 'Above', signal: 'صعودی', distance: '۱.۸%+' },
    { period: 'EMA 50', value: '$41,650', position: 'Above', signal: 'صعودی', distance: '۲.۵%+' }
  ];

  const supportResistance = [
    { level: 'مقاومت ۳', price: '$45,200', distance: '۶.۷%+', strength: 'قوی', touches: 5 },
    { level: 'مقاومت ۲', price: '$43,800', distance: '۳.۴%+', strength: 'متوسط', touches: 3 },
    { level: 'مقاومت ۱', price: '$42,900', distance: '۱.۳%+', strength: 'ضعیف', touches: 2 },
    { level: 'قیمت فعلی', price: '$42,350', distance: '۰%', strength: 'فعلی', touches: 0 },
    { level: 'حمایت ۱', price: '$41,200', distance: '۲.۷%-', strength: 'قوی', touches: 4 },
    { level: 'حمایت ۲', price: '$39,800', distance: '۶.۰%-', strength: 'متوسط', touches: 3 },
    { level: 'حمایت ۳', price: '$38,100', distance: '۱۰.۰%-', strength: 'قوی', touches: 6 }
  ];

  const patterns = [
    { name: 'پرچم صعودی', probability: 75, timeframe: '4H', target: '$44,500', status: 'فعال', confidence: 'بالا' },
    { name: 'مثلث صعودی', probability: 68, timeframe: '1D', target: '$45,200', status: 'در حال شکل‌گیری', confidence: 'متوسط' },
    { name: 'کف دوقلو', probability: 82, timeframe: '1W', target: '$47,800', status: 'تأییدشده', confidence: 'بالا' },
    { name: 'سر و شانه', probability: 45, timeframe: '4H', target: '$39,500', status: 'بالقوه', confidence: 'پایین' }
  ];

  const fibonacciLevels = [
    { level: '۰%', price: '$38,100', type: 'حمایت' },
    { level: '۲۳.۶%', price: '$39,450', type: 'حمایت' },
    { level: '۳۸.۲%', price: '$40,850', type: 'حمایت' },
    { level: '۵۰%', price: '$41,650', type: 'محور' },
    { level: '۶۱.۸%', price: '$42,450', type: 'مقاومت' },
    { level: '۷۸.۶%', price: '$43,250', type: 'مقاومت' },
    { level: '۱۰۰%', price: '$45,200', type: 'مقاومت' }
  ];

  const volumeProfile = [
    { price: '$42,000-42,500', volume: 2450000, percentage: 18.5 },
    { price: '$41,500-42,000', volume: 3200000, percentage: 24.2 },
    { price: '$41,000-41,500', volume: 1850000, percentage: 14.0 },
    { price: '$40,500-41,000', volume: 2100000, percentage: 15.9 },
    { price: '$40,000-40,500', volume: 1650000, percentage: 12.5 }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
      <div className="space-y-6">
        {/* Header with Controls */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold gradient-text">تحلیل تکنیکال</h1>
            <p className="text-gray-400 mt-2">پلتفرم نمودارخوانی حرفه‌ای همراه با تحلیل تکنیکال پیشرفته</p>
          </div>
          <div className="flex items-center space-x-3">
            <Badge className="bg-green-500/20 text-green-300 border-green-500/30">
              <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse" />
              داده زنده
            </Badge>
            <Button onClick={() => setIsFullscreen(!isFullscreen)} className="btn-morphic">
              <Maximize2 className="h-4 w-4 mr-2" />
              تمام‌صفحه
            </Button>
          </div>
        </div>

        {/* Quick Actions - at top, all wired */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
          <Button
            className="btn-liquid text-white h-12"
            onClick={() => {
              setChartRefreshKey((k) => k + 1);
              fetchRealMarketData();
            }}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            به‌روزرسانی داده
          </Button>
          <Button className="btn-morphic h-12" onClick={() => setActivePanel('screener')}>
            <Zap className="h-4 w-4 mr-2" />
            فیلتر نمادها
          </Button>
          <Button className="btn-morphic h-12" onClick={() => setActivePanel('watchlist')}>
            <Target className="h-4 w-4 mr-2" />
            دیده‌بان
          </Button>
          <Button className="btn-morphic h-12" onClick={() => setActivePanel('calendar')}>
            <Calendar className="h-4 w-4 mr-2" />
            تقویم اقتصادی
          </Button>
          <Button className="btn-morphic h-12" onClick={() => setActivePanel('chartSettings')}>
            <Settings className="h-4 w-4 mr-2" />
            تنظیمات نمودار
          </Button>
          <Button className="btn-morphic h-12" onClick={() => setActivePanel('marketOverview')}>
            <Eye className="h-4 w-4 mr-2" />
            نمای کلی بازار
          </Button>
        </div>

        {/* Panels (Sheet) for Screener, Watchlist, Calendar, Chart Settings, Market Overview */}
        <Sheet open={activePanel !== null} onOpenChange={(open) => !open && setActivePanel(null)}>
          <SheetContent side="right" className="w-full sm:max-w-md overflow-y-auto">
            <SheetHeader>
              <SheetTitle>
                {activePanel === 'screener' && 'فیلتر نمادها'}
                {activePanel === 'watchlist' && 'دیده‌بان'}
                {activePanel === 'calendar' && 'تقویم اقتصادی'}
                {activePanel === 'chartSettings' && 'تنظیمات نمودار'}
                {activePanel === 'marketOverview' && 'نمای کلی بازار'}
              </SheetTitle>
            </SheetHeader>
            <div className="mt-6 space-y-4">
              {activePanel === 'screener' && (
                <>
                  <p className="text-sm text-gray-400">فیلتر نمادها بر اساس قیمت، حجم و تکنیکال — متصل به داده زنده از /api/real-market-data.</p>
                  <div className="space-y-2">
                    {['BTC/USD', 'ETH/USD', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'].map((s) => (
                      <Button
                        key={s}
                        variant="outline"
                        className="w-full justify-between"
                        onClick={() => {
                          setSelectedSymbol(s);
                          setActivePanel(null);
                        }}
                      >
                        <span>{s}</span>
                        {realMarketData[SYMBOL_TO_API(s)] && (
                          <span className="text-green-400">${realMarketData[SYMBOL_TO_API(s)].price?.toLocaleString()}</span>
                        )}
                      </Button>
                    ))}
                  </div>
                </>
              )}
              {activePanel === 'watchlist' && (
                <>
                  <p className="text-sm text-gray-400">نمادهای ذخیره‌شده شما (به‌صورت محلی نگهداری می‌شود). برای نمایش در نمودار کلیک کنید.</p>
                  <div className="space-y-2">
                    {(watchlistLoaded ? watchlist : DEFAULT_WATCHLIST).map((s) => (
                      <div key={s} className="flex items-center gap-2">
                        <Button
                          variant="outline"
                          className="flex-1 justify-between"
                          onClick={() => {
                            setSelectedSymbol(s);
                            setActivePanel(null);
                          }}
                        >
                          <Star className="h-4 w-4 fill-current" />
                          <span>{s}</span>
                          {realMarketData[SYMBOL_TO_API(s)] && (
                            <span className="text-green-400 text-xs">${realMarketData[SYMBOL_TO_API(s)].price?.toLocaleString()}</span>
                          )}
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="shrink-0 text-gray-400 hover:text-red-400"
                          onClick={() => removeFromWatchlist(s)}
                          aria-label={`حذف ${s} از دیده‌بان`}
                        >
                          <Minus className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-gray-500 mt-2">نمادها را از فیلتر یا انتخاب‌گر نمودار اضافه کنید.</p>
                </>
              )}
              {activePanel === 'calendar' && (
                <>
                  <p className="text-sm text-gray-400">رویدادهای اقتصادی پیش‌رو از API در صورت در دسترس بودن بک‌اند؛ در غیر این صورت فهرست پیش‌فرض.</p>
                  {calendarLoading ? (
                    <p className="text-sm text-muted-foreground py-4">در حال بارگذاری تقویم…</p>
                  ) : (
                    <ul className="space-y-2 text-sm">
                      {economicEvents.map((ev) => (
                        <li key={`${ev.title}-${ev.date}`} className="flex justify-between items-center p-2 rounded bg-slate-800/50">
                          <span>{ev.title}</span>
                          <div className="flex items-center gap-2">
                            <Badge className="text-xs bg-slate-600/50">{ev.impact}</Badge>
                            <span className="text-gray-400">{ev.date}</span>
                          </div>
                        </li>
                      ))}
                    </ul>
                  )}
                </>
              )}
              {activePanel === 'chartSettings' && (
                <>
                  <p className="text-sm text-gray-400">نوع نمودار و بازه زمانی در کنترل‌های معاملاتی پایین قرار دارند. تمام‌صفحه در هدر است.</p>
                  <div className="space-y-2">
                    <p className="text-xs text-gray-500">نماد: {selectedSymbol}</p>
                    <p className="text-xs text-gray-500">بازه زمانی: {selectedTimeframe}</p>
                  </div>
                </>
              )}
              {activePanel === 'marketOverview' && (
                <>
                  <p className="text-sm text-gray-400">تصویر زنده از API.</p>
                  <div className="space-y-2">
                    {Object.entries(realMarketData).slice(0, 8).map(([sym, d]) => (
                      <div key={sym} className="flex justify-between items-center p-2 rounded bg-slate-800/50 text-sm">
                        <span>{sym}</span>
                        <span className={d.change_percent != null && d.change_percent >= 0 ? 'text-green-400' : 'text-red-400'}>
                          ${d.price?.toLocaleString()} ({d.change_percent != null ? (d.change_percent >= 0 ? '+' : '') + d.change_percent.toFixed(2) : '—'}%)
                        </span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </SheetContent>
        </Sheet>

        {/* Trading Controls */}
        <Card className="glass-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div className="flex items-center space-x-4">
                <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
                  <SelectTrigger className="w-[140px] bg-slate-800/50 border-slate-700">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="BTC/USD">BTC/USD</SelectItem>
                    <SelectItem value="ETH/USD">ETH/USD</SelectItem>
                    <SelectItem value="AAPL">AAPL</SelectItem>
                    <SelectItem value="TSLA">TSLA</SelectItem>
                    <SelectItem value="NVDA">NVDA</SelectItem>
                  </SelectContent>
                </Select>

                <div className="flex space-x-1">
                  {['1m', '5m', '15m', '1H', '4H', '1D', '1W'].map((tf) => (
                    <Button
                      key={tf}
                      onClick={() => setSelectedTimeframe(tf)}
                      className={`px-3 py-1 text-xs ${
                        selectedTimeframe === tf
                          ? 'bg-blue-500 text-white'
                          : 'bg-slate-700/50 text-gray-300 hover:bg-slate-600/50'
                      }`}
                    >
                      {tf}
                    </Button>
                  ))}
                </div>

                <div className="flex space-x-1">
                  {[
                    { type: 'candlestick', icon: BarChart3 },
                    { type: 'line', icon: LineChart },
                    { type: 'area', icon: AreaChart },
                    { type: 'bar', icon: BarChart }
                  ].map(({ type, icon: Icon }) => (
                    <Button
                      key={type}
                      onClick={() => setChartType(type)}
                      className={`p-2 ${
                        chartType === type
                          ? 'bg-blue-500 text-white'
                          : 'bg-slate-700/50 text-gray-300 hover:bg-slate-600/50'
                      }`}
                    >
                      <Icon className="h-4 w-4" />
                    </Button>
                  ))}
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <Button className="btn-morphic text-xs">
                  <Download className="h-3 w-3 mr-1" />
                  خروجی‌گیری
                </Button>
                <Button className="btn-morphic text-xs">
                  <Share2 className="h-3 w-3 mr-1" />
                  اشتراک‌گذاری
                </Button>
                <Button className="btn-morphic text-xs">
                  <Bookmark className="h-3 w-3 mr-1" />
                  ذخیره
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Professional TradingView Chart */}
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* Chart Panel */}
          <div className="xl:col-span-3">
            <TradingViewChart
              key={chartRefreshKey}
              symbol={selectedSymbol}
              interval={selectedTimeframe}
              height={600}
              autoRefresh={true}
            />
          </div>

          {/* Side Panel */}
          <div className="space-y-4">
            {/* Market Info - real data when available */}
            <Card className="glass-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">اطلاعات بازار</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">تغییر ۲۴ساعته</span>
                  <span className={currentMarket?.change_percent != null ? (currentMarket.change_percent >= 0 ? 'text-green-400' : 'text-red-400') : 'text-gray-400'}>
                    {currentMarket?.change_percent != null
                      ? (currentMarket.change_percent >= 0 ? '+' : '') + currentMarket.change_percent.toFixed(2) + '%'
                      : '+2.45%'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">حجم ۲۴ساعته</span>
                  <span>{currentMarket?.volume != null ? (currentMarket.volume / 1e6).toFixed(2) + 'M' : '$2.4B'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">قیمت</span>
                  <span>{currentMarket?.price != null ? '$' + currentMarket.price.toLocaleString() : '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">باز شدن</span>
                  <span>{currentMarket?.open != null ? '$' + currentMarket.open.toLocaleString() : '—'}</span>
                </div>
              </CardContent>
            </Card>

            {/* Quick Indicators */}
            <Card className="glass-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">سیگنال‌های سریع</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {technicalIndicators.slice(0, 4).map((indicator) => (
                  <div key={indicator.name} className="flex items-center justify-between">
                    <div>
                      <div className="text-xs font-medium">{indicator.name}</div>
                      <div className="text-xs text-gray-400">{indicator.recommendation}</div>
                    </div>
                    <Badge className={`text-xs ${
                      indicator.color === 'green' ? 'bg-green-500/20 text-green-300' :
                      indicator.color === 'red' ? 'bg-red-500/20 text-red-300' :
                      indicator.color === 'orange' ? 'bg-orange-500/20 text-orange-300' :
                      'bg-blue-500/20 text-blue-300'
                    }`}>
                      {indicator.signal}
                    </Badge>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Tabbed Analysis */}
        <Tabs defaultValue="indicators" className="w-full">
          <TabsList className="grid w-full grid-cols-6 bg-slate-800/50">
            <TabsTrigger value="indicators">اندیکاتورها</TabsTrigger>
            <TabsTrigger value="patterns">الگوها</TabsTrigger>
            <TabsTrigger value="levels">سطوح</TabsTrigger>
            <TabsTrigger value="fibonacci">فیبوناچی</TabsTrigger>
            <TabsTrigger value="volume">حجم</TabsTrigger>
            <TabsTrigger value="alerts">هشدارها</TabsTrigger>
          </TabsList>

          <TabsContent value="indicators" className="space-y-4">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Gauge className="h-5 w-5 text-blue-400" />
                  <span>اندیکاتورهای تکنیکال</span>
                </CardTitle>
                <CardDescription>اندیکاتورهای جامع تحلیل تکنیکال</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {technicalIndicators.map((indicator, index) => (
                    <div key={indicator.name} className="neomorphic p-4 rounded-lg space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-sm">{indicator.name}</p>
                          <p className="text-xs text-gray-400">{indicator.description}</p>
                        </div>
                        <Badge className={`text-xs ${
                          indicator.color === 'green' ? 'bg-green-500/20 text-green-300' :
                          indicator.color === 'red' ? 'bg-red-500/20 text-red-300' :
                          indicator.color === 'orange' ? 'bg-orange-500/20 text-orange-300' :
                          indicator.color === 'purple' ? 'bg-purple-500/20 text-purple-300' :
                          indicator.color === 'yellow' ? 'bg-yellow-500/20 text-yellow-300' :
                          'bg-blue-500/20 text-blue-300'
                        }`}>
                          {indicator.signal}
                        </Badge>
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-lg font-bold">{indicator.value}</span>
                          <span className="text-xs text-gray-400">{indicator.recommendation}</span>
                        </div>
                        <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div 
                            className={`h-full transition-all duration-1000 ${
                              indicator.color === 'green' ? 'bg-green-400' :
                              indicator.color === 'red' ? 'bg-red-400' :
                              indicator.color === 'orange' ? 'bg-orange-400' :
                              indicator.color === 'purple' ? 'bg-purple-400' :
                              indicator.color === 'yellow' ? 'bg-yellow-400' :
                              'bg-blue-400'
                            }`}
                            style={{ width: `${Math.min(Math.abs(indicator.value), 100)}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="patterns" className="space-y-4">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Activity className="h-5 w-5 text-cyan-400" />
                  <span>الگوهای نمودار</span>
                </CardTitle>
                <CardDescription>الگوهای شناسایی‌شده و اهداف قیمتی</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {patterns.map((pattern, index) => (
                    <div key={pattern.name} className="neomorphic p-6 rounded-lg space-y-4">
                      <div className="text-center">
                        <h3 className="font-bold text-lg">{pattern.name}</h3>
                        <p className="text-sm text-gray-400">{pattern.timeframe} • {pattern.status}</p>
                      </div>
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span className="text-gray-400">احتمال</span>
                          <span className="font-bold text-green-400">{pattern.probability}%</span>
                        </div>
                        <Progress value={pattern.probability} className="h-2" />
                        <div className="flex justify-between">
                          <span className="text-gray-400">هدف</span>
                          <span className="font-bold text-blue-400">{pattern.target}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">اطمینان</span>
                          <Badge className={`text-xs ${
                            pattern.confidence === 'بالا' ? 'bg-green-500/20 text-green-300' :
                            pattern.confidence === 'متوسط' ? 'bg-yellow-500/20 text-yellow-300' :
                            'bg-red-500/20 text-red-300'
                          }`}>
                            {pattern.confidence}
                          </Badge>
                        </div>
                      </div>
                      <Button className="w-full btn-morphic">
                        <Eye className="h-4 w-4 mr-2" />
                        نمایش در نمودار
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="levels" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Support & Resistance */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Target className="h-5 w-5 text-yellow-400" />
                    <span>حمایت و مقاومت</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {supportResistance.map((level, index) => (
                      <div
                        key={level.level}
                        className={`flex items-center justify-between p-3 rounded-lg ${
                          level.level === 'قیمت فعلی'
                            ? 'bg-blue-500/20 border border-blue-500/30'
                            : 'neomorphic'
                        }`}
                      >
                        <div className="flex items-center space-x-4">
                          <span className={`font-medium w-24 text-sm ${
                            level.level.includes('مقاومت') ? 'text-red-400' :
                            level.level.includes('حمایت') ? 'text-green-400' :
                            'text-blue-400'
                          }`}>
                            {level.level}
                          </span>
                          <span className="text-lg font-bold">{level.price}</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <span className={`text-sm ${
                            level.distance.endsWith('+') ? 'text-green-400' :
                            level.distance.endsWith('-') ? 'text-red-400' :
                            'text-gray-400'
                          }`}>
                            {level.distance}
                          </span>
                          <Badge className={`text-xs ${
                            level.strength === 'قوی' ? 'bg-red-500/20 text-red-300' :
                            level.strength === 'متوسط' ? 'bg-yellow-500/20 text-yellow-300' :
                            level.strength === 'فعلی' ? 'bg-blue-500/20 text-blue-300' :
                            'bg-gray-500/20 text-gray-300'
                          }`}>
                            {level.touches} برخورد
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Moving Averages */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <LineChart className="h-5 w-5 text-purple-400" />
                    <span>میانگین‌های متحرک</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {movingAverages.map((ma, index) => (
                      <div key={ma.period} className="flex items-center justify-between p-3 neomorphic rounded-lg">
                        <div className="flex items-center space-x-4">
                          <span className="font-medium w-16 text-sm">{ma.period}</span>
                          <span className="text-lg font-bold">{ma.value}</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <span className="text-sm text-green-400">{ma.distance}</span>
                          <Badge className={`text-xs ${
                            ma.signal === 'صعودی' ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
                          }`}>
                            {ma.signal}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="fibonacci" className="space-y-4">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Layers className="h-5 w-5 text-orange-400" />
                  <span>اصلاح فیبوناچی</span>
                </CardTitle>
                <CardDescription>سطوح کلیدی فیبوناچی برای تحلیل قیمت</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {fibonacciLevels.map((fib, index) => (
                    <div key={fib.level} className="flex items-center justify-between p-3 neomorphic rounded-lg">
                      <div className="flex items-center space-x-4">
                        <span className="font-medium w-16 text-sm">{fib.level}</span>
                        <span className="text-lg font-bold">{fib.price}</span>
                      </div>
                      <Badge className={`text-xs ${
                        fib.type === 'حمایت' ? 'bg-green-500/20 text-green-300' :
                        fib.type === 'مقاومت' ? 'bg-red-500/20 text-red-300' :
                        'bg-blue-500/20 text-blue-300'
                      }`}>
                        {fib.type}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="volume" className="space-y-4">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Volume2 className="h-5 w-5 text-cyan-400" />
                  <span>نمایه حجم</span>
                </CardTitle>
                <CardDescription>توزیع حجم بر اساس سطوح قیمتی</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {volumeProfile.map((vol, index) => (
                    <div key={vol.price} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">{vol.price}</span>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm">{vol.volume.toLocaleString()}</span>
                          <Badge className="bg-cyan-500/20 text-cyan-300 text-xs">
                            {vol.percentage}%
                          </Badge>
                        </div>
                      </div>
                      <div className="w-full h-3 bg-gray-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-cyan-400 transition-all duration-1000"
                          style={{ width: `${vol.percentage}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="alerts" className="space-y-4">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <AlertTriangle className="h-5 w-5 text-yellow-400" />
                  <span>هشدارهای قیمت</span>
                </CardTitle>
                <CardDescription>تنظیم هشدارهای تحلیل تکنیکال</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Button className="btn-morphic h-12">
                    <Target className="h-4 w-4 mr-2" />
                    هشدار قیمت
                  </Button>
                  <Button className="btn-morphic h-12">
                    <Gauge className="h-4 w-4 mr-2" />
                    هشدار اندیکاتور
                  </Button>
                  <Button className="btn-morphic h-12">
                    <Activity className="h-4 w-4 mr-2" />
                    هشدار الگو
                  </Button>
                  <Button className="btn-morphic h-12">
                    <Volume2 className="h-4 w-4 mr-2" />
                    هشدار حجم
                  </Button>
                </div>

                <div className="space-y-3 mt-6">
                  <h4 className="font-medium">هشدارهای فعال</h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between p-3 neomorphic rounded-lg">
                      <div>
                        <p className="font-medium">BTC/USD &gt; $43,000</p>
                        <p className="text-xs text-gray-400">هشدار قیمت</p>
                      </div>
                      <Badge className="bg-green-500/20 text-green-300">فعال</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 neomorphic rounded-lg">
                      <div>
                        <p className="font-medium">RSI &lt; 30</p>
                        <p className="text-xs text-gray-400">هشدار اشباع فروش</p>
                      </div>
                      <Badge className="bg-yellow-500/20 text-yellow-300">در انتظار</Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
} 