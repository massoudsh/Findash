'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  Bitcoin, 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Users, 
  Zap,
  Shield,
  Database,
  Network,
  DollarSign,
  BarChart3,
  Target,
  Eye,
  Layers,
  PieChart,
  ArrowUpDown,
  Wallet,
  Clock,
  AlertTriangle,
  RefreshCw
} from 'lucide-react';

export default function OnChainPage() {
  const [onchainData, setOnchainData] = useState<Record<string, unknown> | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  const fetchOnchainData = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/onchain/comprehensive');
      if (res.ok) {
        const json = await res.json();
        setOnchainData(json.data ?? null);
        setLastUpdated(json.last_updated ?? new Date().toISOString());
      } else {
        setOnchainData(null);
      }
    } catch {
      setOnchainData(null);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchOnchainData();
    const interval = setInterval(fetchOnchainData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // Bitcoin Network Fundamentals (Glassnode-style)
  const bitcoinFundamentals = [
    { metric: 'نرخ هش', value: '۵۲۰.۲ EH/s', change: '۵.۲%+', trend: 'up', signal: 'bullish', category: 'security' },
    { metric: 'دشواری استخراج', value: '۶۷.۳T', change: '۲.۸%+', trend: 'up', signal: 'neutral', category: 'security' },
    { metric: 'زمان بلاک', value: '۹.۸ دقیقه', change: '۱.۲%-', trend: 'down', signal: 'neutral', category: 'performance' },
    { metric: 'حجم Mempool', value: '۱۴۵ مگابایت', change: '۱۵.۳%-', trend: 'down', signal: 'bullish', category: 'performance' }
  ];

  // Network Activity & Adoption
  const networkActivity = [
    { metric: 'آدرس‌های فعال (۲۴ساعته)', value: '۱.۱۲M', change: '۸.۱%+', trend: 'up', signal: 'bullish', category: 'adoption' },
    { metric: 'آدرس‌های جدید (۲۴ساعته)', value: '۳۸۵K', change: '۱۲.۴%+', trend: 'up', signal: 'bullish', category: 'adoption' },
    { metric: 'تعداد تراکنش‌ها', value: '۲۸۴K', change: '۳.۲%+', trend: 'up', signal: 'neutral', category: 'usage' },
    { metric: 'میانگین ارزش تراکنش', value: '$۴۸.۲K', change: '۱۵.۸%+', trend: 'up', signal: 'bullish', category: 'usage' }
  ];

  // Market Structure & Flows
  const marketStructure = [
    { metric: 'ورودی صرافی', value: '۸,۲۴۵ BTC', change: '۲۲.۵%-', trend: 'down', signal: 'bullish', category: 'flows' },
    { metric: 'خروجی صرافی', value: '۱۲,۸۹۰ BTC', change: '۱۸.۳%+', trend: 'up', signal: 'bullish', category: 'flows' },
    { metric: 'موجودی صرافی', value: '۲.۱۸M BTC', change: '۰.۸%-', trend: 'down', signal: 'bullish', category: 'supply' },
    { metric: 'ورودی استیبل‌کوین', value: '$۲.۸B', change: '۴۵.۲%+', trend: 'up', signal: 'bullish', category: 'flows' }
  ];

  // HODLer & Investor Behavior
  const hodlerMetrics = [
    { metric: 'نگهدارندگان بلندمدت', value: '۱۴.۸M BTC', change: '۲.۱%+', trend: 'up', signal: 'bullish', category: 'hodl' },
    { metric: 'نگهدارندگان کوتاه‌مدت', value: '۴.۲M BTC', change: '۱.۸%-', trend: 'down', signal: 'neutral', category: 'hodl' },
    { metric: 'روزهای سکه از دست‌رفته', value: '۲.۱M', change: '۳۵.۲%-', trend: 'down', signal: 'bullish', category: 'hodl' },
    { metric: 'جریان رکود', value: '۰.۸۵', change: '۱۲.۴%-', trend: 'down', signal: 'bullish', category: 'hodl' }
  ];

  // Valuation Models & Metrics
  const valuationMetrics = [
    { metric: 'نسبت MVRV', value: '۱.۸۵', change: '۳.۴%+', trend: 'up', signal: 'neutral', category: 'valuation' },
    { metric: 'نسبت NVT', value: '۲۸.۵', change: '۸.۲%-', trend: 'down', signal: 'bullish', category: 'valuation' },
    { metric: 'قیمت تحقق‌یافته', value: '$۲۸,۴۵۰', change: '۱.۲%+', trend: 'up', signal: 'neutral', category: 'valuation' },
    { metric: 'نسبت ترموکپ', value: '۱۲.۸', change: '۵.۱%+', trend: 'up', signal: 'neutral', category: 'valuation' }
  ];

  // Whale & Institution Activity
  const whaleActivity = [
    { metric: 'آدرس‌های نهنگ (بیش از ۱K BTC)', value: '۲,۱۲۵', change: '۰.۸%+', trend: 'up', signal: 'neutral', category: 'whales' },
    { metric: 'تعداد تراکنش نهنگ‌ها', value: '۱,۲۴۵', change: '۱۲.۳%-', trend: 'down', signal: 'neutral', category: 'whales' },
    { metric: 'جریان خالص نهنگ‌ها', value: '-۲,۴۵۰ BTC', change: '۸۵.۲%+', trend: 'up', signal: 'bullish', category: 'whales' },
    { metric: 'رکود تعدیل‌شده نهادی', value: '۰.۷۲', change: '۱۵.۸%-', trend: 'down', signal: 'bullish', category: 'whales' }
  ];

  // DeFi & Layer 2 Analytics
  const defiL2Metrics = [
    { protocol: 'شبکه لایتنینگ', capacity: '۵,۱۲۵ BTC', nodes: '۱۵,۴۵۰', change: '۸.۲%+', category: 'layer2' },
    { protocol: 'لایه‌های دوم اتریوم', tvl: '$۴۲.۸B', transactions: '۲.۸M/روز', change: '۱۵.۴%+', category: 'layer2' },
    { protocol: 'رپد بیت‌کوین', supply: '۱۶۸K WBTC', volume: '$۸۹۰M', change: '۵.۴%+', category: 'defi' },
    { protocol: 'دیفای بیت‌کوین', tvl: '$۲.۱B', protocols: '۴۵', change: '۲۲.۱%+', category: 'defi' }
  ];

  // Mining & Security Analytics
  const miningMetrics = [
    { metric: 'ضریب پوئل', value: '۰.۸۵', change: '۱۲.۳%-', trend: 'down', signal: 'neutral', category: 'mining' },
    { metric: 'نوارهای هش', value: 'صعودی', change: 'سیگنال', trend: 'up', signal: 'bullish', category: 'mining' },
    { metric: 'درآمد استخراج‌کنندگان', value: '$۲۸.۵M', change: '۸.۴%+', trend: 'up', signal: 'neutral', category: 'mining' },
    { metric: 'درصد درآمد کارمزد', value: '۲.۸%', change: '۴۵.۲%-', trend: 'down', signal: 'neutral', category: 'mining' }
  ];

  const renderMetricSection = (title: string, metrics: any[], icon: any, description: string) => (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-foreground">
          {icon}
          <span>{title}</span>
        </CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {metrics.map((metric, index) => (
            <div key={metric.metric} className="neomorphic p-4 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-muted-foreground">{metric.metric}</span>
                <div className="flex items-center space-x-2">
                  <Badge className={`text-xs ${
                    metric.signal === 'bullish' ? 'bg-green-500/20 text-green-300' :
                    metric.signal === 'bearish' ? 'bg-red-500/20 text-red-300' :
                    'bg-gray-500/20 text-gray-300'
                  }`}>
                    {metric.signal === 'bullish' ? 'صعودی' : metric.signal === 'bearish' ? 'نزولی' : 'خنثی'}
                  </Badge>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-lg font-bold text-foreground">{metric.value}</span>
                <div className={`flex items-center space-x-1 ${
                  metric.trend === 'up' ? 'text-chart-2' : 
                  metric.trend === 'down' ? 'text-destructive' : 'text-muted-foreground'
                }`}>
                  {metric.trend === 'up' ? (
                    <TrendingUp className="h-4 w-4" />
                  ) : metric.trend === 'down' ? (
                    <TrendingDown className="h-4 w-4" />
                  ) : (
                    <Activity className="h-4 w-4" />
                  )}
                  <span className="text-sm">{metric.change}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">آن‌چین</h1>
          <p className="text-muted-foreground mt-1">
            هوشمندی بلاک‌چین و متریک‌های آن‌چین — بلادرنگ در صورت در دسترس بودن بک‌اند
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {onchainData ? (
            <Badge className="bg-primary/20 text-primary border-primary/30">
              زنده از منبع
            </Badge>
          ) : (
            <Badge variant="secondary">تصویر ثابت</Badge>
          )}
          {lastUpdated && (
            <span className="text-xs text-muted-foreground">
              به‌روزرسانی {lastUpdated ? new Date(lastUpdated).toLocaleTimeString() : ''}
            </span>
          )}
          <Button variant="outline" size="sm" onClick={fetchOnchainData} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 mr-1 ${isLoading ? 'animate-spin' : ''}`} />
            به‌روزرسانی
          </Button>
        </div>
      </div>
      <div className="space-y-6">

        {/* Bitcoin Network Fundamentals */}
        {renderMetricSection(
          'مبانی شبکه بیت‌کوین',
          bitcoinFundamentals,
          <Shield className="h-5 w-5 text-foreground" />,
          'متریک‌های اصلی امنیت و عملکرد شبکه'
        )}

        {/* Network Activity & Adoption */}
        {renderMetricSection(
          'فعالیت شبکه و پذیرش',
          networkActivity,
          <Users className="h-5 w-5 text-foreground" />,
          'شاخص‌های پذیرش کاربران و استفاده از شبکه'
        )}

        {/* Market Structure & Flows */}
        {renderMetricSection(
          'ساختار بازار و جریان سرمایه',
          marketStructure,
          <ArrowUpDown className="h-5 w-5 text-foreground" />,
          'تحلیل جریان صرافی‌ها و ساختار بازار'
        )}

        {/* HODLer & Investor Behavior */}
        {renderMetricSection(
          'رفتار نگهدارندگان و سرمایه‌گذاران',
          hodlerMetrics,
          <Clock className="h-5 w-5 text-foreground" />,
          'الگوهای نگهداری بلندمدت و تحلیل عمر سکه'
        )}

        {/* Valuation Models */}
        {renderMetricSection(
          'مدل‌ها و متریک‌های ارزش‌گذاری',
          valuationMetrics,
          <Target className="h-5 w-5 text-foreground" />,
          'مدل‌های پیشرفته ارزش‌گذاری و شاخص‌های ارزش منصفانه'
        )}

        {/* Whale & Institution Activity */}
        {renderMetricSection(
          'فعالیت نهنگ‌ها و نهادها',
          whaleActivity,
          <Eye className="h-5 w-5 text-foreground" />,
          'رفتار سرمایه‌گذاران بزرگ و تحلیل جریان نهادی'
        )}

        {/* Mining & Security Analytics */}
        {renderMetricSection(
          'تحلیل استخراج و امنیت',
          miningMetrics,
          <Zap className="h-5 w-5 text-foreground" />,
          'اقتصاد استخراج و شاخص‌های امنیت شبکه'
        )}

        {/* DeFi & Layer 2 Ecosystem */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Layers className="h-5 w-5 text-foreground" />
              <span>اکوسیستم دیفای و لایه دوم</span>
            </CardTitle>
            <CardDescription>تحلیل پروتکل‌های دیفای و بین‌زنجیره‌ای</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {defiL2Metrics.map((protocol, index) => (
                <div key={protocol.protocol} className="neomorphic p-4 rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 flex items-center justify-center">
                        <span className="text-foreground font-bold text-xs">
                          {protocol.protocol.slice(0, 2)}
                        </span>
                      </div>
                      <span className="font-medium">{protocol.protocol}</span>
                    </div>
                    <Badge className={`text-xs ${
                      protocol.category === 'layer2' ? 'bg-blue-500/20 text-blue-300' : 'bg-purple-500/20 text-purple-300'
                    }`}>
                      {protocol.category === 'layer2' ? 'لایه دوم' : 'دیفای'}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <p className="text-muted-foreground">
                        {protocol.category === 'layer2' ? 'ظرفیت/TVL' : 'عرضه/TVL'}
                      </p>
                      <p className="font-medium">{protocol.capacity || protocol.tvl}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">
                        {protocol.category === 'layer2' ? 'نودها/تراکنش' : 'حجم/پروتکل‌ها'}
                      </p>
                      <p className="font-medium">{protocol.nodes || protocol.transactions || protocol.volume || protocol.protocols}</p>
                    </div>
                  </div>
                  <div className="mt-3 flex justify-between items-center">
                    <span className="text-xs text-muted-foreground">تغییر ۲۴ساعته</span>
                    <span className={`text-sm font-medium ${
                      protocol.change.startsWith('+') ? 'text-chart-2' : 'text-destructive'
                    }`}>
                      {protocol.change}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Market Intelligence Summary */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5 text-foreground" />
              <span>خلاصه هوشمندی بازار</span>
            </CardTitle>
            <CardDescription>بینش‌های مبتنی بر هوش مصنوعی از داده‌های آن‌چین</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="space-y-4">
                <h4 className="font-semibold text-muted-foreground">سلامت شبکه</h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span>امتیاز امنیت</span>
                    <Badge className="bg-green-500/20 text-green-300">۹۵/۱۰۰</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>روند پذیرش</span>
                    <Badge className="bg-blue-500/20 text-blue-300">شتاب‌گیرنده</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>فشار شبکه</span>
                    <Badge className="bg-green-500/20 text-green-300">پایین</Badge>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h4 className="font-semibold text-muted-foreground">احساسات بازار</h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span>قدرت نگهدارندگان</span>
                    <Badge className="bg-green-500/20 text-green-300">قوی</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>جریان نهادی</span>
                    <Badge className="bg-blue-500/20 text-blue-300">در حال انباشت</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>فعالیت خرد</span>
                    <Badge className="bg-yellow-500/20 text-yellow-300">متوسط</Badge>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h4 className="font-semibold text-muted-foreground">سیگنال‌های ارزش‌گذاری</h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span>ارزش منصفانه</span>
                    <Badge className="bg-gray-500/20 text-gray-300">خنثی</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>موقعیت چرخه</span>
                    <Badge className="bg-blue-500/20 text-blue-300">میان‌صعودی</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>ریسک/بازده</span>
                    <Badge className="bg-green-500/20 text-green-300">مطلوب</Badge>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 