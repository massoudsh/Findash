"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  FileText,
  Users,
  DollarSign,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw
} from 'lucide-react';

interface FundamentalSignal {
  signal_type: string;
  strength: number;
  confidence: number;
  description: string;
  contributing_factors: string[];
  timestamp: string;
  expiry?: string;
}

interface FundamentalAnalysis {
  symbol: string;
  timestamp: string;
  asset_type: string;
  signals: FundamentalSignal[];
  summary: {
    overall_sentiment: string;
    key_factors: string[];
    signal_count: number;
    bullish_signals: number;
    bearish_signals: number;
  };
  score: number;
  confidence: number;
}

interface WhaleMetrics {
  large_transactions_24h: number;
  whale_accumulation_score: number;
  exchange_inflow: number;
  exchange_outflow: number;
  net_flow: number;
}

interface FundamentalAnalysisProps {
  symbol: string;
}

const SENTIMENT_LABELS: Record<string, string> = {
  bullish: 'صعودی',
  bearish: 'نزولی',
  neutral: 'خنثی',
  Unknown: 'نامشخص',
};

export function FundamentalAnalysis({ symbol }: FundamentalAnalysisProps) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`/api/fundamental/analysis/${symbol}`);
        const result = await response.json();
        setData(result);
      } catch (error) {
        console.error('Error fetching fundamental data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [symbol]);

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>در حال بارگذاری تحلیل بنیادی...</CardTitle>
        </CardHeader>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>خطا در بارگذاری داده</CardTitle>
        </CardHeader>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>تحلیل بنیادی - {symbol}</CardTitle>
          <CardDescription>امتیاز: {data.score?.toFixed(1) || 'نامشخص'}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-lg font-semibold">{SENTIMENT_LABELS[data.summary?.overall_sentiment] || data.summary?.overall_sentiment || 'نامشخص'}</div>
              <div className="text-sm text-gray-500">احساسات</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold">{data.summary?.signal_count || 0}</div>
              <div className="text-sm text-gray-500">سیگنال‌ها</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold">{((data.confidence || 0) * 100).toFixed(0)}%</div>
              <div className="text-sm text-gray-500">اطمینان</div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="signals">
        <TabsList>
          <TabsTrigger value="signals">سیگنال‌ها</TabsTrigger>
          <TabsTrigger value="onchain">آنچین</TabsTrigger>
          <TabsTrigger value="whale">فعالیت نهنگ‌ها</TabsTrigger>
        </TabsList>

        <TabsContent value="signals">
          <Card>
            <CardHeader>
              <CardTitle>سیگنال‌های بنیادی</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {data.signals?.map((signal: any, index: number) => (
                  <div key={index} className="p-3 border rounded-lg">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-medium">{signal.signal_type}</div>
                        <div className="text-sm text-gray-600">{signal.description}</div>
                      </div>
                      <Badge>{(signal.confidence * 100).toFixed(0)}%</Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="onchain">
          <Card>
            <CardHeader>
              <CardTitle>معیارهای آنچین</CardTitle>
              <CardDescription>فعالیت شبکه بلاک‌چین و شاخص‌های سلامت</CardDescription>
            </CardHeader>
            <CardContent>
              {data.on_chain_metrics ? (
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="text-sm text-gray-500">آدرس‌های فعال (۲۴ ساعت)</div>
                    <div className="text-lg font-semibold">{data.on_chain_metrics.active_addresses_24h?.toLocaleString() || 'نامشخص'}</div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-sm text-gray-500">حجم تراکنش (۲۴ ساعت)</div>
                    <div className="text-lg font-semibold">${(data.on_chain_metrics.transaction_volume_24h / 1e9)?.toFixed(2) || 'نامشخص'}B</div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-sm text-gray-500">نسبت MVRV</div>
                    <div className="text-lg font-semibold">{data.on_chain_metrics.mvrv_ratio?.toFixed(2) || 'نامشخص'}</div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-sm text-gray-500">نسبت NVT</div>
                    <div className="text-lg font-semibold">{data.on_chain_metrics.network_value_to_transactions?.toFixed(1) || 'نامشخص'}</div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-sm text-gray-500">هش‌ریت</div>
                    <div className="text-lg font-semibold">{data.on_chain_metrics.hash_rate || 'نامشخص'}</div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-sm text-gray-500">تعدیل دشواری</div>
                    <div className="text-lg font-semibold">{data.on_chain_metrics.difficulty_adjustment || 'نامشخص'}</div>
                  </div>
                </div>
              ) : (
                <p className="text-gray-500">داده آنچین برای {symbol} در دسترس نیست</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="whale">
          <Card>
            <CardHeader>
              <CardTitle>فعالیت نهنگ‌ها</CardTitle>
              <CardDescription>الگوهای تراکنش و جریان دارندگان بزرگ</CardDescription>
            </CardHeader>
            <CardContent>
              {data.whale_metrics ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="text-sm text-gray-500">تراکنش‌های بزرگ (۲۴ ساعت)</div>
                      <div className="text-lg font-semibold">{data.whale_metrics.large_transactions_24h || 'نامشخص'}</div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm text-gray-500">امتیاز انباشت</div>
                      <div className="text-lg font-semibold">{data.whale_metrics.whale_accumulation_score?.toFixed(1) || 'نامشخص'}/۱۰</div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <h4 className="font-medium">جریان‌های صرافی</h4>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="text-center p-3 bg-red-50 rounded-lg">
                        <div className="text-red-600 font-semibold">{data.whale_metrics.exchange_inflow?.toLocaleString() || 'نامشخص'}</div>
                        <div className="text-xs text-red-500">ورودی</div>
                      </div>
                      <div className="text-center p-3 bg-green-50 rounded-lg">
                        <div className="text-green-600 font-semibold">{data.whale_metrics.exchange_outflow?.toLocaleString() || 'نامشخص'}</div>
                        <div className="text-xs text-green-500">خروجی</div>
                      </div>
                      <div className="text-center p-3 bg-blue-50 rounded-lg">
                        <div className={`font-semibold ${data.whale_metrics.net_flow < 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {data.whale_metrics.net_flow?.toLocaleString() || 'نامشخص'}
                        </div>
                        <div className="text-xs text-blue-500">جریان خالص</div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-gray-500">داده فعالیت نهنگ‌ها برای {symbol} در دسترس نیست</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
