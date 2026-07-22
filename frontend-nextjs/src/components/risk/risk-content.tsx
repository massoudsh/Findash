'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  AlertTriangle,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
  BarChart3,
  PieChart,
  Target,
  Zap,
  Activity,
  RefreshCw,
  Settings,
  Calculator,
  DollarSign,
  Percent,
  Timer,
  Brain
} from 'lucide-react';

// Enhanced interfaces for skfolio-inspired metrics
interface SkfolioRiskMetrics {
  // Basic Risk Metrics
  portfolioValue: number;
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;

  // Performance Metrics
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  omega_ratio: number;

  // Risk-Return Metrics
  max_drawdown: number;
  volatility: number;
  skewness: number;
  kurtosis: number;

  // Portfolio Construction Metrics
  diversification_ratio: number;
  effective_number_assets: number;
  concentration_risk: number;
  turnover: number;

  // Risk Budgeting
  risk_contribution: Record<string, number>;
  marginal_var: Record<string, number>;
  component_var: Record<string, number>;

  // Factor Exposures
  factor_loadings: Record<string, number>;
  factor_var_decomposition: Record<string, number>;

  // Tail Risk
  tail_ratio: number;
  gain_loss_ratio: number;
  pain_index: number;
  ulcer_index: number;

  // Optimization Metrics
  risk_parity_distance: number;
  mean_variance_efficiency: number;
  black_litterman_views: Record<string, number>;
}

interface OptimizationTarget {
  objective: 'max_sharpe' | 'min_variance' | 'risk_parity' | 'max_diversification' | 'black_litterman';
  constraints: {
    max_weight?: number;
    min_weight?: number;
    sector_constraints?: Record<string, number>;
    turnover_constraint?: number;
    leverage_constraint?: number;
  };
  rebalancing_frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly';
}

interface RebalancingRecommendation {
  current_weights: Record<string, number>;
  target_weights: Record<string, number>;
  trades_required: Array<{
    symbol: string;
    action: 'buy' | 'sell';
    quantity: number;
    value: number;
    reason: string;
  }>;
  expected_improvement: {
    sharpe_delta: number;
    var_delta: number;
    diversification_delta: number;
  };
  implementation_cost: number;
}

export function RiskContent() {
  const [metrics, setMetrics] = useState<SkfolioRiskMetrics | null>(null);
  const [rebalancingRec, setRebalancingRec] = useState<RebalancingRecommendation | null>(null);
  const [optimizationTarget, setOptimizationTarget] = useState<OptimizationTarget>({
    objective: 'max_sharpe',
    constraints: {
      max_weight: 0.15,
      min_weight: 0.01,
      turnover_constraint: 0.20
    },
    rebalancing_frequency: 'monthly'
  });
  const [isLoading, setIsLoading] = useState(true);
  const [isOptimizing, setIsOptimizing] = useState(false);

  useEffect(() => {
    fetchRiskMetrics();
  }, []);

  const fetchRiskMetrics = async () => {
    setIsLoading(true);
    try {
      // Simulate API call to skfolio backend
      const response = await fetch('/api/risk/skfolio-metrics');
      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
      } else {
        // Fallback to mock data
        setMetrics(generateMockSkfolioMetrics());
      }
    } catch (error) {
      console.error('Failed to fetch skfolio metrics:', error);
      setMetrics(generateMockSkfolioMetrics());
    } finally {
      setIsLoading(false);
    }
  };

  const runOptimization = async () => {
    setIsOptimizing(true);
    try {
      const response = await fetch('/api/risk/optimize-portfolio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(optimizationTarget)
      });

      if (response.ok) {
        const recommendation = await response.json();
        setRebalancingRec(recommendation);
      } else {
        // Fallback to mock recommendation
        setRebalancingRec(generateMockRebalancingRec());
      }
    } catch (error) {
      console.error('Optimization failed:', error);
      setRebalancingRec(generateMockRebalancingRec());
    } finally {
      setIsOptimizing(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2">در حال بارگذاری تحلیل ریسک...</span>
      </div>
    );
  }

  if (!metrics) {
    return <div className="text-center text-red-600">بارگذاری معیارهای ریسک ممکن نشد.</div>;
  }

  return (
    <div className="space-y-6">
      {/* Risk Overview Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="glass-card">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ارزش پورتفولیو</CardTitle>
            <DollarSign className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">${metrics.portfolioValue.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">ارزش کل بازار</p>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">نسبت شارپ</CardTitle>
            <Target className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.sharpe_ratio.toFixed(2)}</div>
            <p className="text-xs text-muted-foreground">بازده تعدیل‌شده با ریسک</p>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ارزش در معرض ریسک (۹۵٪)</CardTitle>
            <AlertTriangle className="h-4 w-4 text-orange-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">${metrics.var_95.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">زیان احتمالی ۱ روزه</p>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">تنوع‌بخشی</CardTitle>
            <PieChart className="h-4 w-4 text-purple-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.diversification_ratio.toFixed(2)}</div>
            <p className="text-xs text-muted-foreground">تنوع پورتفولیو</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="metrics" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="metrics">معیارهای ریسک</TabsTrigger>
          <TabsTrigger value="optimization">بهینه‌سازی</TabsTrigger>
          <TabsTrigger value="risk-budgeting">بودجه‌بندی ریسک</TabsTrigger>
          <TabsTrigger value="factor-analysis">تحلیل عوامل</TabsTrigger>
        </TabsList>

        <TabsContent value="metrics" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            {/* Performance Metrics */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  معیارهای عملکرد
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-sm text-muted-foreground">نسبت شارپ</Label>
                    <div className="text-lg font-semibold">{metrics.sharpe_ratio.toFixed(3)}</div>
                  </div>
                  <div>
                    <Label className="text-sm text-muted-foreground">نسبت سورتینو</Label>
                    <div className="text-lg font-semibold">{metrics.sortino_ratio.toFixed(3)}</div>
                  </div>
                  <div>
                    <Label className="text-sm text-muted-foreground">نسبت کالمار</Label>
                    <div className="text-lg font-semibold">{metrics.calmar_ratio.toFixed(3)}</div>
                  </div>
                  <div>
                    <Label className="text-sm text-muted-foreground">نسبت اُمگا</Label>
                    <div className="text-lg font-semibold">{metrics.omega_ratio.toFixed(3)}</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Risk Metrics */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ShieldCheck className="h-5 w-5" />
                  معیارهای ریسک
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-sm text-muted-foreground">ارزش در معرض ریسک ۹۵٪</Label>
                    <div className="text-lg font-semibold text-red-600">${metrics.var_95.toLocaleString()}</div>
                  </div>
                  <div>
                    <Label className="text-sm text-muted-foreground">CVaR ۹۵٪</Label>
                    <div className="text-lg font-semibold text-red-600">${metrics.cvar_95.toLocaleString()}</div>
                  </div>
                  <div>
                    <Label className="text-sm text-muted-foreground">حداکثر افت سرمایه</Label>
                    <div className="text-lg font-semibold text-red-600">{(metrics.max_drawdown * 100).toFixed(1)}%</div>
                  </div>
                  <div>
                    <Label className="text-sm text-muted-foreground">نوسان‌پذیری</Label>
                    <div className="text-lg font-semibold">{(metrics.volatility * 100).toFixed(1)}%</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Tail Risk Metrics */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingDown className="h-5 w-5" />
                  تحلیل ریسک دنباله
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-sm text-muted-foreground">چولگی</Label>
                    <div className="text-lg font-semibold">{metrics.skewness.toFixed(3)}</div>
                  </div>
                  <div>
                    <Label className="text-sm text-muted-foreground">کشیدگی</Label>
                    <div className="text-lg font-semibold">{metrics.kurtosis.toFixed(3)}</div>
                  </div>
                  <div>
                    <Label className="text-sm text-muted-foreground">نسبت دنباله</Label>
                    <div className="text-lg font-semibold">{metrics.tail_ratio.toFixed(3)}</div>
                  </div>
                  <div>
                    <Label className="text-sm text-muted-foreground">شاخص اولسر</Label>
                    <div className="text-lg font-semibold">{metrics.ulcer_index.toFixed(3)}</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Portfolio Construction */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  ساختار پورتفولیو
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-sm text-muted-foreground">تعداد مؤثر دارایی‌ها</Label>
                    <div className="text-lg font-semibold">{metrics.effective_number_assets.toFixed(1)}</div>
                  </div>
                  <div>
                    <Label className="text-sm text-muted-foreground">ریسک تمرکز</Label>
                    <div className="text-lg font-semibold">{(metrics.concentration_risk * 100).toFixed(1)}%</div>
                  </div>
                  <div>
                    <Label className="text-sm text-muted-foreground">گردش معاملات</Label>
                    <div className="text-lg font-semibold">{(metrics.turnover * 100).toFixed(1)}%</div>
                  </div>
                  <div>
                    <Label className="text-sm text-muted-foreground">فاصله برابری ریسک</Label>
                    <div className="text-lg font-semibold">{metrics.risk_parity_distance.toFixed(3)}</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="optimization" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            {/* Optimization Settings */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  بهینه‌سازی پورتفولیو
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>هدف بهینه‌سازی</Label>
                  <Select
                    value={optimizationTarget.objective}
                    onValueChange={(value: any) => setOptimizationTarget(prev => ({ ...prev, objective: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="max_sharpe">حداکثر‌سازی نسبت شارپ</SelectItem>
                      <SelectItem value="min_variance">حداقل‌سازی واریانس</SelectItem>
                      <SelectItem value="risk_parity">برابری ریسک</SelectItem>
                      <SelectItem value="max_diversification">حداکثر تنوع‌بخشی</SelectItem>
                      <SelectItem value="black_litterman">بلک-لیترمن</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>حداکثر وزن (%)</Label>
                    <Input
                      type="number"
                      value={(optimizationTarget.constraints.max_weight || 0) * 100}
                      onChange={(e) => setOptimizationTarget(prev => ({
                        ...prev,
                        constraints: { ...prev.constraints, max_weight: parseFloat(e.target.value) / 100 }
                      }))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>حداقل وزن (%)</Label>
                    <Input
                      type="number"
                      value={(optimizationTarget.constraints.min_weight || 0) * 100}
                      onChange={(e) => setOptimizationTarget(prev => ({
                        ...prev,
                        constraints: { ...prev.constraints, min_weight: parseFloat(e.target.value) / 100 }
                      }))}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>دوره تعادل‌بخشی مجدد</Label>
                  <Select
                    value={optimizationTarget.rebalancing_frequency}
                    onValueChange={(value: any) => setOptimizationTarget(prev => ({ ...prev, rebalancing_frequency: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="daily">روزانه</SelectItem>
                      <SelectItem value="weekly">هفتگی</SelectItem>
                      <SelectItem value="monthly">ماهانه</SelectItem>
                      <SelectItem value="quarterly">فصلی</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button
                  onClick={runOptimization}
                  disabled={isOptimizing}
                  className="w-full"
                >
                  {isOptimizing ? (
                    <>
                      <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                      در حال بهینه‌سازی...
                    </>
                  ) : (
                    <>
                      <Zap className="mr-2 h-4 w-4" />
                      اجرای بهینه‌سازی
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Rebalancing Recommendations */}
            {rebalancingRec && (
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <RefreshCw className="h-5 w-5" />
                    پیشنهادهای تعادل‌بخشی مجدد
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-sm text-muted-foreground">Δ شارپ</div>
                      <div className={`text-lg font-semibold ${rebalancingRec.expected_improvement.sharpe_delta > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {rebalancingRec.expected_improvement.sharpe_delta > 0 ? '+' : ''}
                        {rebalancingRec.expected_improvement.sharpe_delta.toFixed(3)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Δ ارزش در معرض ریسک</div>
                      <div className={`text-lg font-semibold ${rebalancingRec.expected_improvement.var_delta < 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {rebalancingRec.expected_improvement.var_delta > 0 ? '+' : ''}
                        ${rebalancingRec.expected_improvement.var_delta.toLocaleString()}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Δ تنوع‌بخشی</div>
                      <div className={`text-lg font-semibold ${rebalancingRec.expected_improvement.diversification_delta > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {rebalancingRec.expected_improvement.diversification_delta > 0 ? '+' : ''}
                        {rebalancingRec.expected_improvement.diversification_delta.toFixed(3)}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>معاملات مورد نیاز</Label>
                    <div className="max-h-32 overflow-y-auto space-y-1">
                      {rebalancingRec.trades_required.map((trade, index) => (
                        <div key={index} className="flex justify-between items-center p-2 bg-muted rounded">
                          <span className="font-medium">{trade.symbol}</span>
                          <Badge variant={trade.action === 'buy' ? 'default' : 'destructive'}>
                            {trade.action === 'buy' ? 'خرید' : 'فروش'} ${trade.value.toLocaleString()}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="text-sm text-muted-foreground">
                    هزینه اجرا: ${rebalancingRec.implementation_cost.toLocaleString()}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="risk-budgeting" className="space-y-4">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calculator className="h-5 w-5" />
                تحلیل سهم ریسک
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(metrics.risk_contribution).map(([asset, contribution]) => (
                  <div key={asset} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">{asset}</span>
                      <span className="text-sm text-muted-foreground">
                        {(contribution * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={contribution * 100} className="h-2" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="factor-analysis" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  بارهای عاملی
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {Object.entries(metrics.factor_loadings).map(([factor, loading]) => (
                    <div key={factor} className="flex justify-between items-center">
                      <span className="font-medium">{factor}</span>
                      <span className={`font-semibold ${loading > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {loading > 0 ? '+' : ''}{loading.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <PieChart className="h-5 w-5" />
                  تجزیه ارزش در معرض ریسک بر اساس عوامل
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(metrics.factor_var_decomposition).map(([factor, var_contrib]) => (
                    <div key={factor} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="font-medium">{factor}</span>
                        <span className="text-sm text-muted-foreground">
                          ${var_contrib.toLocaleString()}
                        </span>
                      </div>
                      <Progress value={(var_contrib / metrics.var_95) * 100} className="h-2" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {/* Risk Alerts */}
      <Alert>
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          <strong>هشدار ریسک:</strong> ریسک تمرکز پورتفولیو بالاتر از سطح بهینه است.
          برای بهبود تنوع‌بخشی، تعادل‌بخشی مجدد را در نظر بگیرید.
        </AlertDescription>
      </Alert>
    </div>
  );
}

// Mock data generators
function generateMockSkfolioMetrics(): SkfolioRiskMetrics {
  return {
    portfolioValue: 1250000,
    var_95: 18750,
    var_99: 31250,
    cvar_95: 25600,
    cvar_99: 42800,
    sharpe_ratio: 1.847,
    sortino_ratio: 2.234,
    calmar_ratio: 1.456,
    omega_ratio: 1.678,
    max_drawdown: 0.087,
    volatility: 0.156,
    skewness: -0.234,
    kurtosis: 3.456,
    diversification_ratio: 0.834,
    effective_number_assets: 12.4,
    concentration_risk: 0.234,
    turnover: 0.045,
    risk_contribution: {
      'AAPL': 0.156,
      'MSFT': 0.134,
      'GOOGL': 0.123,
      'AMZN': 0.145,
      'TSLA': 0.089,
      'NVDA': 0.098,
      'Others': 0.255
    },
    marginal_var: {
      'AAPL': 2340,
      'MSFT': 2150,
      'GOOGL': 2890,
      'AMZN': 2670,
      'TSLA': 3450,
      'NVDA': 3120
    },
    component_var: {
      'AAPL': 2925,
      'MSFT': 2513,
      'GOOGL': 2306,
      'AMZN': 2719,
      'TSLA': 1669,
      'NVDA': 1838
    },
    factor_loadings: {
      'Market': 0.856,
      'Size': -0.123,
      'Value': 0.234,
      'Momentum': 0.456,
      'Quality': 0.345,
      'Low Vol': -0.234
    },
    factor_var_decomposition: {
      'Market': 12450,
      'Size': 1230,
      'Value': 2340,
      'Momentum': 1890,
      'Quality': 1560,
      'Idiosyncratic': 2280
    },
    tail_ratio: 0.456,
    gain_loss_ratio: 1.234,
    pain_index: 0.123,
    ulcer_index: 0.089,
    risk_parity_distance: 0.234,
    mean_variance_efficiency: 0.789,
    black_litterman_views: {
      'AAPL': 0.08,
      'MSFT': 0.12,
      'GOOGL': 0.06,
      'AMZN': 0.10
    }
  };
}

function generateMockRebalancingRec(): RebalancingRecommendation {
  return {
    current_weights: {
      'AAPL': 0.156,
      'MSFT': 0.134,
      'GOOGL': 0.123,
      'AMZN': 0.145,
      'TSLA': 0.089,
      'NVDA': 0.098
    },
    target_weights: {
      'AAPL': 0.140,
      'MSFT': 0.145,
      'GOOGL': 0.135,
      'AMZN': 0.130,
      'TSLA': 0.075,
      'NVDA': 0.110
    },
    trades_required: [
      { symbol: 'AAPL', action: 'sell', quantity: 45, value: 8100, reason: 'کاهش تمرکز' },
      { symbol: 'MSFT', action: 'buy', quantity: 28, value: 9240, reason: 'افزایش تخصیص' },
      { symbol: 'GOOGL', action: 'buy', quantity: 12, value: 3360, reason: 'تعادل‌بخشی به سمت هدف' },
      { symbol: 'TSLA', action: 'sell', quantity: 18, value: 3780, reason: 'کاهش نوسان‌پذیری' },
      { symbol: 'NVDA', action: 'buy', quantity: 15, value: 4200, reason: 'افزایش قرارگیری در حوزه فناوری' }
    ],
    expected_improvement: {
      sharpe_delta: 0.043,
      var_delta: -1250,
      diversification_delta: 0.023
    },
    implementation_cost: 450
  };
}
