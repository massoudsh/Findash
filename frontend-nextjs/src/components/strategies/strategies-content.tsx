'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from '@/components/ui/toast';
import { formatCurrency, formatPercentage } from '@/lib/utils';
import { getStrategies, createStrategy } from '@/lib/services/api';
import { getStrategyPerformance, type Strategy } from '@/lib/services/strategies_api';
import { StrategyMiniChart } from '@/components/strategies/strategy-mini-chart';
import { Plus, Play, TrendingUp, Settings, DollarSign, Target, Clock, AlertTriangle, Info, BarChart3, Activity, Shield, Zap, TrendingDown, Layers, ExternalLink } from 'lucide-react';
import Link from 'next/link';

interface NewStrategyForm {
  name: string;
  description: string;
  strategy_type: string;
  symbols: string;
  initial_capital: number;
  risk_budget: number;
  max_drawdown_limit: number;
  target_sharpe: number;
  rebalance_frequency: string;
  // Strategy-specific parameters
  rsi_period: number;
  rsi_oversold: number;
  rsi_overbought: number;
  bb_period: number;
  bb_stddev: number;
  momentum_lookback: number;
  momentum_threshold: number;
  volatility_threshold: number;
  time_horizon: string;
  active: boolean;
}

function NewStrategyForm({ onSuccess }: { onSuccess?: () => void }): JSX.Element {
  const [form, setForm] = useState<NewStrategyForm>({
    name: '',
    description: '',
    strategy_type: '',
    symbols: 'AAPL,TSLA,MSFT,GOOGL,AMZN',
    initial_capital: 100000,
    risk_budget: 0.05,
    max_drawdown_limit: 0.15,
    target_sharpe: 1.0,
    rebalance_frequency: 'monthly',
    rsi_period: 14,
    rsi_oversold: 30,
    rsi_overbought: 70,
    bb_period: 20,
    bb_stddev: 2,
    momentum_lookback: 20,
    momentum_threshold: 0.02,
    volatility_threshold: 0.1,
    time_horizon: '1d',
    active: true,
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState<Partial<NewStrategyForm>>({});

  const strategyTypes = [
    { value: 'momentum', label: 'استراتژی مومنتوم', description: 'روند و مومنتوم قیمت را دنبال می‌کند' },
    { value: 'technical', label: 'تحلیل تکنیکال', description: 'از RSI، باندهای بولینگر و سایر شاخص‌ها استفاده می‌کند' },
    { value: 'risk_aware', label: 'استراتژی ریسک‌محور', description: 'بر مدیریت ریسک و کنترل افت سرمایه تمرکز دارد' },
    { value: 'mean_reversion', label: 'بازگشت به میانگین', description: 'بر خلاف نقاط اوج و کف قیمت معامله می‌کند' },
    { value: 'breakout', label: 'استراتژی شکست قیمتی', description: 'بر اساس شکست قیمت از محدوده معامله می‌کند' },
    { value: 'volatility_spread', label: 'اسپرد نوسان', description: 'استراتژی اختیار معامله بر اساس اسپرد نوسان ضمنی/تاریخی' },
  ];

  const rebalanceFrequencies = [
    { value: 'daily', label: 'روزانه' },
    { value: 'weekly', label: 'هفتگی' },
    { value: 'monthly', label: 'ماهانه' },
    { value: 'quarterly', label: 'فصلی' },
  ];

  const timeHorizons = [
    { value: '1m', label: '۱ دقیقه' },
    { value: '5m', label: '۵ دقیقه' },
    { value: '15m', label: '۱۵ دقیقه' },
    { value: '1h', label: '۱ ساعت' },
    { value: '4h', label: '۴ ساعت' },
    { value: '1d', label: '۱ روز' },
    { value: '1w', label: '۱ هفته' },
  ];

  function handleInputChange(e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) {
    const { name, value, type } = e.target;
    let processedValue: any = value;

    if (type === 'number') {
      processedValue = value === '' ? 0 : parseFloat(value);
      if (isNaN(processedValue)) processedValue = 0;
    }

    setForm((prev) => ({
      ...prev,
      [name]: processedValue
    }));

    // Clear error when user starts typing
    if (errors[name as keyof NewStrategyForm]) {
      setErrors((prev) => ({ ...prev, [name]: undefined }));
    }
  }

  function handleSelectChange(name: keyof NewStrategyForm, value: string) {
    setForm((prev) => ({ ...prev, [name]: value }));
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: undefined }));
    }
  }

  function handleCheckboxChange(e: React.ChangeEvent<HTMLInputElement>) {
    const { name, checked } = e.target;
    setForm((prev) => ({ ...prev, [name]: checked }));
  }

  function validateForm(): boolean {
    const newErrors: Partial<NewStrategyForm> = {};

    if (!form.name.trim()) newErrors.name = 'نام استراتژی الزامی است';
    if (!form.description.trim()) newErrors.description = 'توضیحات الزامی است';
    if (!form.strategy_type) newErrors.strategy_type = 'نوع استراتژی الزامی است';
    if (!form.symbols.trim()) newErrors.symbols = 'حداقل یک نماد الزامی است';
    if (Number(form.initial_capital) <= 0) newErrors.initial_capital = 'سرمایه اولیه باید مثبت باشد' as any;
    if (Number(form.risk_budget) <= 0 || Number(form.risk_budget) > 1) newErrors.risk_budget = 'بودجه ریسک باید بین ۰ و ۱ باشد' as any;
    if (Number(form.max_drawdown_limit) <= 0 || Number(form.max_drawdown_limit) > 1) newErrors.max_drawdown_limit = 'حداکثر افت سرمایه باید بین ۰ و ۱ باشد' as any;
    if (Number(form.target_sharpe) <= 0) newErrors.target_sharpe = 'نسبت شارپ هدف باید مثبت باشد' as any;

    // Strategy-specific validations
    if (form.strategy_type === 'technical' || form.strategy_type === 'momentum') {
      if (Number(form.rsi_period) <= 0) newErrors.rsi_period = 'دوره RSI باید مثبت باشد' as any;
      if (Number(form.rsi_oversold) <= 0 || Number(form.rsi_oversold) >= 50) newErrors.rsi_oversold = 'سطح اشباع فروش RSI باید بین ۰ و ۵۰ باشد' as any;
      if (Number(form.rsi_overbought) <= 50 || Number(form.rsi_overbought) >= 100) newErrors.rsi_overbought = 'سطح اشباع خرید RSI باید بین ۵۰ و ۱۰۰ باشد' as any;
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!validateForm()) return;

    setIsSubmitting(true);
    try {
      // Prepare strategy parameters based on type
      const strategyParams: Record<string, any> = {};

      if (form.strategy_type === 'technical') {
        strategyParams.rsi_period = form.rsi_period;
        strategyParams.bb_period = form.bb_period;
        strategyParams.bb_stddev = form.bb_stddev;
      } else if (form.strategy_type === 'momentum') {
        strategyParams.lookback_period = form.momentum_lookback;
        strategyParams.threshold = form.momentum_threshold;
      } else if (form.strategy_type === 'risk_aware') {
        strategyParams.max_drawdown = form.max_drawdown_limit;
        strategyParams.var_confidence = 0.95;
      } else if (form.strategy_type === 'volatility_spread') {
        strategyParams.volatility_spread_threshold = form.volatility_threshold;
      }

      // Create strategy payload
      const strategyData = {
        name: form.name,
        description: form.description,
        strategy_type: form.strategy_type,
        symbols: form.symbols.split(',').map(s => s.trim()).filter(Boolean),
        initial_capital: form.initial_capital,
        parameters: {
          ...strategyParams,
          risk_budget: form.risk_budget,
          max_drawdown_limit: form.max_drawdown_limit,
          target_sharpe: form.target_sharpe,
          rebalance_frequency: form.rebalance_frequency,
          time_horizon: form.time_horizon,
        },
        is_active: form.active,
      };

      await createStrategy(strategyData);

      toast({
        title: 'استراتژی ایجاد شد',
        description: `${form.name} با موفقیت ایجاد شد.`,
      });

      onSuccess?.();

      // Reset form
      setForm({
        name: '',
        description: '',
        strategy_type: '',
        symbols: 'AAPL,TSLA,MSFT,GOOGL,AMZN',
        initial_capital: 100000,
        risk_budget: 0.05,
        max_drawdown_limit: 0.15,
        target_sharpe: 1.0,
        rebalance_frequency: 'monthly',
        rsi_period: 14,
        rsi_oversold: 30,
        rsi_overbought: 70,
        bb_period: 20,
        bb_stddev: 2,
        momentum_lookback: 20,
        momentum_threshold: 0.02,
        volatility_threshold: 0.1,
        time_horizon: '1d',
        active: true,
      });

      onSuccess?.();
    } catch (error) {
      toast({
        title: 'خطا',
        description: 'ایجاد استراتژی ناموفق بود. دوباره تلاش کنید.',
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  const selectedStrategyType = strategyTypes.find(st => st.value === form.strategy_type);

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Basic Information */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Settings className="h-4 w-4" />
          <h3 className="text-lg font-medium">اطلاعات پایه</h3>
        </div>

        <div className="grid grid-cols-1 gap-4">
          <div>
            <Label htmlFor="name">نام استراتژی *</Label>
            <Input
              id="name"
              name="name"
              value={form.name}
              onChange={handleInputChange}
              placeholder="مثلاً استراتژی مومنتوم من"
              className={errors.name ? 'border-red-500' : ''}
            />
            {errors.name && <p className="text-sm text-red-500 mt-1">{errors.name}</p>}
          </div>

          <div>
            <Label htmlFor="description">توضیحات *</Label>
            <Textarea
              id="description"
              name="description"
              value={form.description}
              onChange={handleInputChange}
              placeholder="رویکرد و اهداف استراتژی خود را شرح دهید..."
              className={errors.description ? 'border-red-500' : ''}
              rows={3}
            />
            {errors.description && <p className="text-sm text-red-500 mt-1">{errors.description}</p>}
          </div>

          <div>
            <Label htmlFor="strategy_type">نوع استراتژی *</Label>
            <Select value={form.strategy_type} onValueChange={(value) => handleSelectChange('strategy_type', value)}>
              <SelectTrigger className={errors.strategy_type ? 'border-red-500' : ''}>
                <SelectValue placeholder="نوع استراتژی را انتخاب کنید" />
              </SelectTrigger>
              <SelectContent>
                {strategyTypes.map((type) => (
                  <SelectItem key={type.value} value={type.value}>
                    <div>
                      <div className="font-medium">{type.label}</div>
                      <div className="text-sm text-gray-500">{type.description}</div>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {errors.strategy_type && <p className="text-sm text-red-500 mt-1">{errors.strategy_type}</p>}
            {selectedStrategyType && (
              <p className="text-sm text-gray-600 mt-1">{selectedStrategyType.description}</p>
            )}
          </div>
        </div>
      </div>

      {/* Portfolio Settings */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <DollarSign className="h-4 w-4" />
          <h3 className="text-lg font-medium">تنظیمات پورتفولیو</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <Label htmlFor="symbols">نمادها (جدا شده با کاما) *</Label>
            <Input
              id="symbols"
              name="symbols"
              value={form.symbols}
              onChange={handleInputChange}
              placeholder="AAPL,TSLA,MSFT,GOOGL,AMZN"
              className={errors.symbols ? 'border-red-500' : ''}
            />
            {errors.symbols && <p className="text-sm text-red-500 mt-1">{errors.symbols}</p>}
          </div>

          <div>
            <Label htmlFor="initial_capital">سرمایه اولیه ($) *</Label>
            <Input
              id="initial_capital"
              name="initial_capital"
              type="number"
              value={form.initial_capital}
              onChange={handleInputChange}
              min="1000"
              step="1000"
              className={errors.initial_capital ? 'border-red-500' : ''}
            />
            {errors.initial_capital && <p className="text-sm text-red-500 mt-1">{errors.initial_capital}</p>}
          </div>

          <div>
            <Label htmlFor="rebalance_frequency">دوره تعادل‌بخشی مجدد</Label>
            <Select value={form.rebalance_frequency} onValueChange={(value) => handleSelectChange('rebalance_frequency', value)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {rebalanceFrequencies.map((freq) => (
                  <SelectItem key={freq.value} value={freq.value}>
                    {freq.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label htmlFor="time_horizon">بازه زمانی</Label>
            <Select value={form.time_horizon} onValueChange={(value) => handleSelectChange('time_horizon', value)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {timeHorizons.map((horizon) => (
                  <SelectItem key={horizon.value} value={horizon.value}>
                    {horizon.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Risk Management */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-4 w-4" />
          <h3 className="text-lg font-medium">مدیریت ریسک</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <Label htmlFor="risk_budget">بودجه ریسک (۰ تا ۱) *</Label>
            <Input
              id="risk_budget"
              name="risk_budget"
              type="number"
              value={form.risk_budget}
              onChange={handleInputChange}
              min="0.001"
              max="1"
              step="0.001"
              className={errors.risk_budget ? 'border-red-500' : ''}
            />
            {errors.risk_budget && <p className="text-sm text-red-500 mt-1">{errors.risk_budget}</p>}
            <p className="text-xs text-gray-500 mt-1">حداکثر ریسک پورتفولیو (۵٪ = 0.05)</p>
          </div>

          <div>
            <Label htmlFor="max_drawdown_limit">حداکثر افت سرمایه (۰ تا ۱) *</Label>
            <Input
              id="max_drawdown_limit"
              name="max_drawdown_limit"
              type="number"
              value={form.max_drawdown_limit}
              onChange={handleInputChange}
              min="0.01"
              max="1"
              step="0.01"
              className={errors.max_drawdown_limit ? 'border-red-500' : ''}
            />
            {errors.max_drawdown_limit && <p className="text-sm text-red-500 mt-1">{errors.max_drawdown_limit}</p>}
            <p className="text-xs text-gray-500 mt-1">حداکثر افت سرمایه قابل قبول (۱۵٪ = 0.15)</p>
          </div>

          <div>
            <Label htmlFor="target_sharpe">نسبت شارپ هدف *</Label>
            <Input
              id="target_sharpe"
              name="target_sharpe"
              type="number"
              value={form.target_sharpe}
              onChange={handleInputChange}
              min="0.1"
              step="0.1"
              className={errors.target_sharpe ? 'border-red-500' : ''}
            />
            {errors.target_sharpe && <p className="text-sm text-red-500 mt-1">{errors.target_sharpe}</p>}
            <p className="text-xs text-gray-500 mt-1">هدف بازده تعدیل‌شده با ریسک</p>
          </div>
        </div>
      </div>

      {/* Strategy-Specific Parameters */}
      {(form.strategy_type === 'technical' || form.strategy_type === 'momentum') && (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            <h3 className="text-lg font-medium">پارامترهای تکنیکال</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Label htmlFor="rsi_period">دوره RSI</Label>
              <Input
                id="rsi_period"
                name="rsi_period"
                type="number"
                value={form.rsi_period}
                onChange={handleInputChange}
                min="2"
                max="50"
                className={errors.rsi_period ? 'border-red-500' : ''}
              />
              {errors.rsi_period && <p className="text-sm text-red-500 mt-1">{errors.rsi_period}</p>}
            </div>

            <div>
              <Label htmlFor="rsi_oversold">سطح اشباع فروش RSI</Label>
              <Input
                id="rsi_oversold"
                name="rsi_oversold"
                type="number"
                value={form.rsi_oversold}
                onChange={handleInputChange}
                min="10"
                max="40"
                className={errors.rsi_oversold ? 'border-red-500' : ''}
              />
              {errors.rsi_oversold && <p className="text-sm text-red-500 mt-1">{errors.rsi_oversold}</p>}
            </div>

            <div>
              <Label htmlFor="rsi_overbought">سطح اشباع خرید RSI</Label>
              <Input
                id="rsi_overbought"
                name="rsi_overbought"
                type="number"
                value={form.rsi_overbought}
                onChange={handleInputChange}
                min="60"
                max="90"
                className={errors.rsi_overbought ? 'border-red-500' : ''}
              />
              {errors.rsi_overbought && <p className="text-sm text-red-500 mt-1">{errors.rsi_overbought}</p>}
            </div>

            {form.strategy_type === 'technical' && (
              <>
                <div>
                  <Label htmlFor="bb_period">دوره باندهای بولینگر</Label>
                  <Input
                    id="bb_period"
                    name="bb_period"
                    type="number"
                    value={form.bb_period}
                    onChange={handleInputChange}
                    min="5"
                    max="50"
                  />
                </div>

                <div>
                  <Label htmlFor="bb_stddev">انحراف معیار باندهای بولینگر</Label>
                  <Input
                    id="bb_stddev"
                    name="bb_stddev"
                    type="number"
                    value={form.bb_stddev}
                    onChange={handleInputChange}
                    min="1"
                    max="3"
                    step="0.1"
                  />
                </div>
              </>
            )}

            {form.strategy_type === 'momentum' && (
              <>
                <div>
                  <Label htmlFor="momentum_lookback">دوره بازنگری مومنتوم</Label>
                  <Input
                    id="momentum_lookback"
                    name="momentum_lookback"
                    type="number"
                    value={form.momentum_lookback}
                    onChange={handleInputChange}
                    min="5"
                    max="100"
                  />
                </div>

                <div>
                  <Label htmlFor="momentum_threshold">آستانه مومنتوم</Label>
                  <Input
                    id="momentum_threshold"
                    name="momentum_threshold"
                    type="number"
                    value={form.momentum_threshold}
                    onChange={handleInputChange}
                    min="0.001"
                    max="0.1"
                    step="0.001"
                  />
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {form.strategy_type === 'volatility_spread' && (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            <h3 className="text-lg font-medium">پارامترهای نوسان</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="volatility_threshold">آستانه اسپرد نوسان</Label>
              <Input
                id="volatility_threshold"
                name="volatility_threshold"
                type="number"
                value={form.volatility_threshold}
                onChange={handleInputChange}
                min="0.01"
                max="0.5"
                step="0.01"
              />
              <p className="text-xs text-gray-500 mt-1">حداقل اسپرد نوسان ضمنی-تاریخی برای ایجاد سیگنال</p>
            </div>
          </div>
        </div>
      )}

      {/* Status */}
      <div className="flex items-center space-x-2">
        <input
          id="active"
          name="active"
          type="checkbox"
          checked={form.active}
          onChange={handleCheckboxChange}
          className="rounded border-gray-300"
        />
        <Label htmlFor="active">فعال‌سازی فوری استراتژی</Label>
      </div>

      {/* Submit Button */}
      <div className="flex gap-3 pt-4">
        <Button type="submit" disabled={isSubmitting} className="flex-1">
          {isSubmitting ? 'در حال ایجاد...' : 'ایجاد استراتژی'}
        </Button>
      </div>
    </form>
  );
}

// Strategy Details Modal Component
function StrategyDetailsModal({ strategy, isOpen, onClose }: {
  strategy: Strategy | null;
  isOpen: boolean;
  onClose: () => void;
}) {
  if (!strategy) return null;

  const getStrategyIcon = (type: string) => {
    switch (type) {
      case 'momentum':
        return <TrendingUp className="h-5 w-5" />;
      case 'technical':
        return <BarChart3 className="h-5 w-5" />;
      case 'risk_aware':
        return <Shield className="h-5 w-5" />;
      case 'mean_reversion':
        return <TrendingDown className="h-5 w-5" />;
      case 'breakout':
        return <Zap className="h-5 w-5" />;
      case 'volatility_spread':
        return <Activity className="h-5 w-5" />;
      default:
        return <Layers className="h-5 w-5" />;
    }
  };

  const getStrategyDescription = (type: string) => {
    switch (type) {
      case 'momentum':
        return {
          overview: 'استراتژی‌های مومنتوم از تمایل دارایی‌ها به ادامه حرکت در جهت فعلی خود بهره می‌برند. این استراتژی روندها را شناسایی کرده و برای کسب سود آن‌ها را دنبال می‌کند.',
          methodology: 'از میانگین‌های متحرک، شاخص‌های مومنتوم قیمت و معیارهای قدرت روند برای شناسایی و پیگیری روندهای بازار استفاده می‌کند.',
          riskProfile: 'متوسط تا بالا - می‌تواند در بازگشت روندها افت سرمایه قابل‌توجهی را تجربه کند',
          bestMarkets: 'بازارهای روند‌دار، بازارهای صعودی، بازارهای رمزارز با حرکات جهت‌دار قوی',
          parameters: ['دوره بازنگری: ۲۰ روز', 'آستانه مومنتوم: ۲٪', 'حد ضرر: ۵٪', 'اندازه پوزیشن: ۱۰ تا ۲۵٪'],
          pros: ['ثبت حرکات بزرگ بازار', 'عملکرد خوب در بازارهای روند‌دار', 'قابلیت تولید بازده بالا'],
          cons: ['آسیب‌پذیر در برابر نوسانات کاذب', 'عملکرد ضعیف در بازارهای خنثی', 'ورود و خروج با تأخیر']
        };
      case 'technical':
        return {
          overview: 'استراتژی تحلیل تکنیکال با استفاده از چندین شاخص شامل RSI، باندهای بولینگر، MACD و میانگین‌های متحرک برای شناسایی نقاط ورود و خروج.',
          methodology: 'چندین شاخص تکنیکال را برای تولید سیگنال‌های با اطمینان بالا ترکیب می‌کند. از RSI برای مومنتوم، باندهای بولینگر برای نوسان و میانگین‌های متحرک برای روند استفاده می‌کند.',
          riskProfile: 'متوسط - منابع سیگنال متنوع، سیگنال‌های کاذب را کاهش می‌دهد',
          bestMarkets: 'همه شرایط بازار، به‌ویژه مؤثر در بازارهای پرنوسان',
          parameters: ['دوره RSI: ۱۴', 'اشباع فروش RSI: ۳۰', 'اشباع خرید RSI: ۷۰', 'دوره باند بولینگر: ۲۰', 'انحراف معیار باند بولینگر: ۲'],
          pros: ['سیگنال‌های تأییدی متعدد', 'عملکرد در شرایط مختلف بازار', 'شاخص‌های آزموده‌شده'],
          cons: ['ممکن است در واکنش کند باشد', 'مستعد سیگنال‌های کاذب در بازارهای پرنوسان کوتاه‌مدت', 'نیاز به بهینه‌سازی پارامتر']
        };
      case 'risk_aware':
        return {
          overview: 'استراتژی ریسک‌محور که با مدیریت فعال ریسک، اندازه‌گیری پوزیشن و کنترل افت سرمایه، حفظ سرمایه را در اولویت قرار می‌دهد.',
          methodology: 'از ارزش در معرض ریسک (VaR)، محدودیت‌های حداکثر افت سرمایه و اندازه‌گیری پویای پوزیشن برای کنترل ریسک پورتفولیو ضمن جستجوی بازده پایدار استفاده می‌کند.',
          riskProfile: 'پایین تا متوسط - تمرکز اصلی بر حفظ سرمایه',
          bestMarkets: 'همه شرایط بازار، به‌ویژه بازارهای پرنوسان یا نامطمئن',
          parameters: ['حداکثر افت سرمایه: ۱۵٪', 'اطمینان VaR: ۹۵٪', 'بودجه ریسک: ۵٪', 'محدودیت پوزیشن: ۱۰٪'],
          pros: ['تمرکز بر حفظ سرمایه', 'بازده پایدار تعدیل‌شده با ریسک', 'نوسان‌پذیری کمتر'],
          cons: ['ممکن است حرکات صعودی بزرگ را از دست بدهد', 'بازده مطلق کمتر', 'رویکرد محافظه‌کارانه']
        };
      case 'mean_reversion':
        return {
          overview: 'استراتژی بازگشت به میانگین فرض می‌کند که قیمت‌ها در طول زمان به میانگین تاریخی خود بازمی‌گردند و از نقاط اوج و کف قیمت سود می‌برد.',
          methodology: 'شرایط اشباع خرید و فروش را با استفاده از معیارهای آماری شناسایی کرده و برخلاف روند غالب معامله می‌کند.',
          riskProfile: 'متوسط - می‌تواند دوره‌های طولانی زیان تحقق‌نیافته را تجربه کند',
          bestMarkets: 'بازارهای محدود به یک بازه، معاملات پرتناوب، معاملات جفتی',
          parameters: ['دوره بازنگری: ۳۰ روز', 'آستانه Z-Score: ۲.۰', 'دوره نگهداری: ۵ تا ۱۰ روز'],
          pros: ['سود از ناکارآمدی‌های بازار', 'عملکرد خوب در بازارهای خنثی', 'سطوح حد ضرر طبیعی'],
          cons: ['عملکرد ضعیف در بازارهای روند‌دار', 'زمان‌بندی حیاتی است', 'ممکن است در روند نزولی گرفتار شود']
        };
      case 'breakout':
        return {
          overview: 'استراتژی شکست قیمتی زمانی را شناسایی می‌کند که قیمت از سطوح مهم حمایت یا مقاومت عبور کند و روندهای جدید احتمالی را نشان دهد.',
          methodology: 'محدوده‌های قیمتی، الگوهای حجم و نوسان را برای تشخیص شکست‌های واقعی از سیگنال‌های کاذب رصد می‌کند.',
          riskProfile: 'متوسط تا بالا - شناسایی زودهنگام روند اما مستعد شکست‌های کاذب',
          bestMarkets: 'بازارهایی که از تثبیت به فاز روند‌دار در حال گذار هستند',
          parameters: ['آستانه شکست: ۲٪', 'تأیید حجم: ۱۵۰٪', 'دوره محدوده: ۲۰ روز'],
          pros: ['شناسایی زودهنگام روند', 'سیگنال‌های ورود واضح', 'نسبت‌های ریسک به بازده قوی'],
          cons: ['شکست‌های کاذب زیاد', 'نیاز به اجرای سریع', 'هزینه‌های معاملاتی بالا']
        };
      case 'volatility_spread':
        return {
          overview: 'استراتژی اختیار معامله که از اختلاف بین نوسان ضمنی (IV) و نوسان تاریخی (HV) سود می‌برد.',
          methodology: 'نوسان ضمنی فعلی را با نوسان تاریخی مقایسه می‌کند تا اختیار معامله‌های نادرست قیمت‌گذاری‌شده و فرصت‌های نوسان را شناسایی کند.',
          riskProfile: 'متوسط - محدود به پرمیوم اختیار معامله و کاهش ارزش زمانی',
          bestMarkets: 'بازارهای اختیار معامله با انحراف نوسان بالا، فصل‌های گزارش‌دهی مالی',
          parameters: ['آستانه IV-HV: ۱۰٪', 'روز تا سررسید: ۳۰ تا ۴۵', 'محدوده دلتا: ۰.۳ تا ۰.۷'],
          pros: ['سود از قیمت‌گذاری نادرست نوسان', 'قرارگیری محدود در معرض ریسک', 'پتانسیل خنثی از بازار'],
          cons: ['نیاز به دانش اختیار معامله', 'ریسک کاهش ارزش زمانی', 'محدودیت‌های نقدشوندگی']
        };
      default:
        return {
          overview: 'استراتژی معاملاتی سفارشی با پارامترها و قوانین مدیریت ریسک مشخص.',
          methodology: 'از الگوریتم‌ها و شاخص‌های اختصاصی برای تولید سیگنال‌های معاملاتی استفاده می‌کند.',
          riskProfile: 'متغیر بر اساس پیکربندی استراتژی',
          bestMarkets: 'بستگی به طراحی استراتژی و پارامترها دارد',
          parameters: ['پارامترهای سفارشی بر اساس نوع استراتژی'],
          pros: ['متناسب با شرایط خاص بازار', 'پارامترهای قابل تنظیم'],
          cons: ['نیاز به بک‌تست', 'ممکن است نیاز به بهینه‌سازی داشته باشد']
        };
    }
  };

  const strategyInfo = getStrategyDescription(strategy.strategy_type);

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center gap-3">
            {getStrategyIcon(strategy.strategy_type)}
            <div>
              <DialogTitle className="text-xl">{strategy.name}</DialogTitle>
              <DialogDescription className="text-base">
                {strategy.description}
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">نمای کلی</TabsTrigger>
            <TabsTrigger value="methodology">روش‌شناسی</TabsTrigger>
            <TabsTrigger value="parameters">پارامترها</TabsTrigger>
            <TabsTrigger value="performance">عملکرد</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Info className="h-4 w-4" />
                  نمای کلی استراتژی
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-gray-600">{strategyInfo.overview}</p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-sm mb-2">نمایه ریسک</h4>
                    <Badge variant="outline">{strategyInfo.riskProfile}</Badge>
                  </div>
                  <div>
                    <h4 className="font-medium text-sm mb-2">بهترین بازارها</h4>
                    <p className="text-sm text-gray-600">{strategyInfo.bestMarkets}</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-sm mb-2 text-green-600">مزایا</h4>
                    <ul className="text-sm text-gray-600 space-y-1">
                      {strategyInfo.pros.map((pro, index) => (
                        <li key={index} className="flex items-start gap-2">
                          <span className="text-green-500 mt-1">•</span>
                          {pro}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-sm mb-2 text-red-600">ملاحظات</h4>
                    <ul className="text-sm text-gray-600 space-y-1">
                      {strategyInfo.cons.map((con, index) => (
                        <li key={index} className="flex items-start gap-2">
                          <span className="text-red-500 mt-1">•</span>
                          {con}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="methodology" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>روش‌شناسی استراتژی</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-600 leading-relaxed">{strategyInfo.methodology}</p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="parameters" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>پارامترهای استراتژی</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {strategyInfo.parameters.map((param, index) => (
                    <div key={index} className="flex items-center gap-2 p-2 bg-gray-50 rounded">
                      <Target className="h-3 w-3 text-gray-500" />
                      <span className="text-sm font-mono">{param}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="performance" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>معیارهای عملکرد</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-gray-50 rounded">
                    <div className="text-lg font-bold text-gray-900">--</div>
                    <div className="text-xs text-gray-500">بازده کل</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded">
                    <div className="text-lg font-bold text-gray-900">--</div>
                    <div className="text-xs text-gray-500">نسبت شارپ</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded">
                    <div className="text-lg font-bold text-gray-900">--</div>
                    <div className="text-xs text-gray-500">حداکثر افت سرمایه</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded">
                    <div className="text-lg font-bold text-gray-900">--</div>
                    <div className="text-xs text-gray-500">نرخ برد</div>
                  </div>
                </div>
                <p className="text-sm text-gray-500 mt-4">
                  معیارهای عملکرد پس از اجرای بک‌تست در دسترس خواهند بود.
                </p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}

export function StrategiesContent() {
  const router = useRouter();
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showNewStrategyModal, setShowNewStrategyModal] = useState(false);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);

  useEffect(() => {
    async function fetchStrategies() {
      try {
        const response = await getStrategies();
        setStrategies(response.data);
      } catch (error) {
        console.error('Error fetching strategies:', error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchStrategies();
  }, []);

  async function handleNewStrategySuccess() {
    setShowNewStrategyModal(false);
    try {
      const { data } = await getStrategies();
      setStrategies(Array.isArray(data) ? data : []);
    } catch (e) {
      console.error('Error refetching strategies:', e);
    }
  }

  function handleViewDetails(strategy: Strategy) {
    setSelectedStrategy(strategy);
    setShowDetailsModal(true);
  }

  function handleBacktest(strategy: Strategy) {
    // Navigate to Strategies > Backtesting tab with strategy pre-filled
    const params = new URLSearchParams({
      tab: 'backtesting',
      strategy_type: strategy.strategy_type,
      strategy_name: strategy.name,
      symbols: strategy.symbols?.join(',') || 'AAPL,TSLA,MSFT',
      initial_capital: strategy.initial_capital?.toString() || '100000'
    });
    router.push(`/strategies?${params.toString()}`);
  }

  if (isLoading) {
    return <div>در حال بارگذاری استراتژی‌ها...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
        <div>
          <h2 className="text-lg font-semibold">استراتژی‌های فعال</h2>
          <p className="text-sm text-muted-foreground">
            استراتژی‌های معاملاتی خود را مدیریت و پایش کنید
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <Button variant="outline" size="sm" asChild>
            <Link href="/options?tab=strategies" className="flex items-center gap-1">
              <Activity className="h-4 w-4" />
              استراتژی‌های اختیار معامله
            </Link>
          </Button>
          <Button onClick={() => setShowNewStrategyModal(true)}>
            <Plus className="h-4 w-4 mr-2" />
            استراتژی جدید
          </Button>
        </div>
      </div>

      {/* New Strategy Modal */}
      <Dialog open={showNewStrategyModal} onOpenChange={setShowNewStrategyModal}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>ایجاد استراتژی معاملاتی جدید</DialogTitle>
            <DialogDescription>
              یک استراتژی معاملاتی جدید با پارامترهای سفارشی و تنظیمات مدیریت ریسک پیکربندی کنید.
            </DialogDescription>
          </DialogHeader>
          <NewStrategyForm onSuccess={handleNewStrategySuccess} />
        </DialogContent>
      </Dialog>

      {/* Strategy Details Modal */}
      <StrategyDetailsModal
        strategy={selectedStrategy}
        isOpen={showDetailsModal}
        onClose={() => setShowDetailsModal(false)}
      />

      {strategies.length === 0 ? (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center py-8">
              <TrendingUp className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium mb-2">استراتژی‌ای یافت نشد</h3>
              <p className="text-muted-foreground mb-4">
                برای شروع، اولین استراتژی معاملاتی خود را ایجاد کنید
              </p>
              <Button onClick={() => setShowNewStrategyModal(true)}>
                <Plus className="h-4 w-4 mr-2" />
                ایجاد استراتژی
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {strategies.map((strategy) => (
            <Card key={strategy.id} className="bg-gray-900 border-gray-800">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg text-white flex items-center gap-2">
                    {strategy.strategy_type === 'momentum' && <TrendingUp className="h-4 w-4" />}
                    {strategy.strategy_type === 'technical' && <BarChart3 className="h-4 w-4" />}
                    {strategy.strategy_type === 'risk_aware' && <Shield className="h-4 w-4" />}
                    {strategy.strategy_type === 'mean_reversion' && <TrendingDown className="h-4 w-4" />}
                    {strategy.strategy_type === 'breakout' && <Zap className="h-4 w-4" />}
                    {strategy.strategy_type === 'volatility_spread' && <Activity className="h-4 w-4" />}
                    {strategy.name}
                  </CardTitle>
                  <div className={`px-2 py-1 rounded-full text-xs ${
                    strategy.is_active
                      ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                      : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
                  }`}>
                    {strategy.is_active ? 'فعال' : 'غیرفعال'}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <StrategyMiniChart strategyType={strategy.strategy_type} height={72} className="rounded" />
                  <p className="text-sm text-gray-300 line-clamp-2">
                    {strategy.description}
                  </p>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">نوع:</span>
                    <span className="font-medium capitalize text-gray-200">{strategy.strategy_type.replace('_', ' ')}</span>
                  </div>
                  <div className="flex flex-wrap gap-2 pt-2">
                    <Button
                      size="sm"
                      variant="outline"
                      className="flex-1 min-w-0"
                      onClick={() => handleBacktest(strategy)}
                    >
                      <Play className="h-3 w-3 mr-1 shrink-0" />
                      بک‌تست
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="flex-1 min-w-0"
                      onClick={() => handleViewDetails(strategy)}
                    >
                      مشاهده جزئیات
                    </Button>
                    {strategy.strategy_type === 'volatility_spread' && (
                      <Button size="sm" variant="outline" className="w-full" asChild>
                        <Link href="/options?tab=strategies">
                          <ExternalLink className="h-3 w-3 mr-1 shrink-0" />
                          باز کردن در اختیار معامله
                        </Link>
                      </Button>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
