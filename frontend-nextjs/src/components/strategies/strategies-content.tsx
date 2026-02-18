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
    { value: 'momentum', label: 'Momentum Strategy', description: 'Follows price trends and momentum' },
    { value: 'technical', label: 'Technical Analysis', description: 'Uses RSI, Bollinger Bands, and other indicators' },
    { value: 'risk_aware', label: 'Risk-Aware Strategy', description: 'Focuses on risk management and drawdown control' },
    { value: 'mean_reversion', label: 'Mean Reversion', description: 'Trades against price extremes' },
    { value: 'breakout', label: 'Breakout Strategy', description: 'Trades on price breakouts from ranges' },
    { value: 'volatility_spread', label: 'Volatility Spread', description: 'Options strategy based on IV/HV spread' },
  ];

  const rebalanceFrequencies = [
    { value: 'daily', label: 'Daily' },
    { value: 'weekly', label: 'Weekly' },
    { value: 'monthly', label: 'Monthly' },
    { value: 'quarterly', label: 'Quarterly' },
  ];

  const timeHorizons = [
    { value: '1m', label: '1 Minute' },
    { value: '5m', label: '5 Minutes' },
    { value: '15m', label: '15 Minutes' },
    { value: '1h', label: '1 Hour' },
    { value: '4h', label: '4 Hours' },
    { value: '1d', label: '1 Day' },
    { value: '1w', label: '1 Week' },
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

    if (!form.name.trim()) newErrors.name = 'Strategy name is required';
    if (!form.description.trim()) newErrors.description = 'Description is required';
    if (!form.strategy_type) newErrors.strategy_type = 'Strategy type is required';
    if (!form.symbols.trim()) newErrors.symbols = 'At least one symbol is required';
    if (Number(form.initial_capital) <= 0) newErrors.initial_capital = 'Initial capital must be positive' as any;
    if (Number(form.risk_budget) <= 0 || Number(form.risk_budget) > 1) newErrors.risk_budget = 'Risk budget must be between 0 and 1' as any;
    if (Number(form.max_drawdown_limit) <= 0 || Number(form.max_drawdown_limit) > 1) newErrors.max_drawdown_limit = 'Max drawdown must be between 0 and 1' as any;
    if (Number(form.target_sharpe) <= 0) newErrors.target_sharpe = 'Target Sharpe ratio must be positive' as any;

    // Strategy-specific validations
    if (form.strategy_type === 'technical' || form.strategy_type === 'momentum') {
      if (Number(form.rsi_period) <= 0) newErrors.rsi_period = 'RSI period must be positive' as any;
      if (Number(form.rsi_oversold) <= 0 || Number(form.rsi_oversold) >= 50) newErrors.rsi_oversold = 'RSI oversold must be between 0 and 50' as any;
      if (Number(form.rsi_overbought) <= 50 || Number(form.rsi_overbought) >= 100) newErrors.rsi_overbought = 'RSI overbought must be between 50 and 100' as any;
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
        title: 'Strategy Created',
        description: `${form.name} has been created successfully.`,
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
        title: 'Error',
        description: 'Failed to create strategy. Please try again.',
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
          <h3 className="text-lg font-medium">Basic Information</h3>
        </div>
        
        <div className="grid grid-cols-1 gap-4">
          <div>
            <Label htmlFor="name">Strategy Name *</Label>
            <Input
              id="name"
              name="name"
              value={form.name}
              onChange={handleInputChange}
              placeholder="e.g., My Momentum Strategy"
              className={errors.name ? 'border-red-500' : ''}
            />
            {errors.name && <p className="text-sm text-red-500 mt-1">{errors.name}</p>}
          </div>

          <div>
            <Label htmlFor="description">Description *</Label>
            <Textarea
              id="description"
              name="description"
              value={form.description}
              onChange={handleInputChange}
              placeholder="Describe your strategy's approach and goals..."
              className={errors.description ? 'border-red-500' : ''}
              rows={3}
            />
            {errors.description && <p className="text-sm text-red-500 mt-1">{errors.description}</p>}
          </div>

          <div>
            <Label htmlFor="strategy_type">Strategy Type *</Label>
            <Select value={form.strategy_type} onValueChange={(value) => handleSelectChange('strategy_type', value)}>
              <SelectTrigger className={errors.strategy_type ? 'border-red-500' : ''}>
                <SelectValue placeholder="Select strategy type" />
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
          <h3 className="text-lg font-medium">Portfolio Settings</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <Label htmlFor="symbols">Symbols (comma-separated) *</Label>
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
            <Label htmlFor="initial_capital">Initial Capital ($) *</Label>
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
            <Label htmlFor="rebalance_frequency">Rebalance Frequency</Label>
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
            <Label htmlFor="time_horizon">Time Horizon</Label>
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
          <h3 className="text-lg font-medium">Risk Management</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <Label htmlFor="risk_budget">Risk Budget (0-1) *</Label>
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
            <p className="text-xs text-gray-500 mt-1">Maximum portfolio risk (5% = 0.05)</p>
          </div>

          <div>
            <Label htmlFor="max_drawdown_limit">Max Drawdown (0-1) *</Label>
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
            <p className="text-xs text-gray-500 mt-1">Maximum acceptable drawdown (15% = 0.15)</p>
          </div>

          <div>
            <Label htmlFor="target_sharpe">Target Sharpe Ratio *</Label>
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
            <p className="text-xs text-gray-500 mt-1">Risk-adjusted return target</p>
          </div>
        </div>
      </div>

      {/* Strategy-Specific Parameters */}
      {(form.strategy_type === 'technical' || form.strategy_type === 'momentum') && (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            <h3 className="text-lg font-medium">Technical Parameters</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Label htmlFor="rsi_period">RSI Period</Label>
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
              <Label htmlFor="rsi_oversold">RSI Oversold Level</Label>
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
              <Label htmlFor="rsi_overbought">RSI Overbought Level</Label>
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
                  <Label htmlFor="bb_period">Bollinger Bands Period</Label>
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
                  <Label htmlFor="bb_stddev">Bollinger Bands Std Dev</Label>
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
                  <Label htmlFor="momentum_lookback">Momentum Lookback Period</Label>
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
                  <Label htmlFor="momentum_threshold">Momentum Threshold</Label>
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
            <h3 className="text-lg font-medium">Volatility Parameters</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="volatility_threshold">Volatility Spread Threshold</Label>
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
              <p className="text-xs text-gray-500 mt-1">Minimum IV-HV spread to trigger signal</p>
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
        <Label htmlFor="active">Activate strategy immediately</Label>
      </div>

      {/* Submit Button */}
      <div className="flex gap-3 pt-4">
        <Button type="submit" disabled={isSubmitting} className="flex-1">
          {isSubmitting ? 'Creating...' : 'Create Strategy'}
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
          overview: 'Momentum strategies capitalize on the tendency of assets to continue moving in their current direction. This strategy identifies trends and rides them for profit.',
          methodology: 'Uses moving averages, price momentum indicators, and trend strength metrics to identify and follow market trends.',
          riskProfile: 'Medium to High - Can experience significant drawdowns during trend reversals',
          bestMarkets: 'Trending markets, bull markets, crypto markets with strong directional moves',
          parameters: ['Lookback Period: 20 days', 'Momentum Threshold: 2%', 'Stop Loss: 5%', 'Position Size: 10-25%'],
          pros: ['Captures large market moves', 'Works well in trending markets', 'Can generate high returns'],
          cons: ['Vulnerable to whipsaws', 'Poor performance in sideways markets', 'Late entries and exits']
        };
      case 'technical':
        return {
          overview: 'Technical analysis strategy using multiple indicators including RSI, Bollinger Bands, MACD, and moving averages to identify entry and exit points.',
          methodology: 'Combines multiple technical indicators to generate high-confidence signals. Uses RSI for momentum, Bollinger Bands for volatility, and moving averages for trend.',
          riskProfile: 'Medium - Diversified signal sources reduce false signals',
          bestMarkets: 'All market conditions, particularly effective in volatile markets',
          parameters: ['RSI Period: 14', 'RSI Oversold: 30', 'RSI Overbought: 70', 'BB Period: 20', 'BB Std Dev: 2'],
          pros: ['Multiple confirmation signals', 'Works in various market conditions', 'Well-tested indicators'],
          cons: ['Can be slow to react', 'Prone to false signals in choppy markets', 'Requires parameter optimization']
        };
      case 'risk_aware':
        return {
          overview: 'Risk-first strategy that prioritizes capital preservation through active risk management, position sizing, and drawdown control.',
          methodology: 'Uses Value at Risk (VaR), maximum drawdown limits, and dynamic position sizing to control portfolio risk while seeking consistent returns.',
          riskProfile: 'Low to Medium - Primary focus on capital preservation',
          bestMarkets: 'All market conditions, especially volatile or uncertain markets',
          parameters: ['Max Drawdown: 15%', 'VaR Confidence: 95%', 'Risk Budget: 5%', 'Position Limit: 10%'],
          pros: ['Capital preservation focus', 'Consistent risk-adjusted returns', 'Lower volatility'],
          cons: ['May miss large upside moves', 'Lower absolute returns', 'Conservative approach']
        };
      case 'mean_reversion':
        return {
          overview: 'Mean reversion strategy assumes that prices will return to their historical average over time, profiting from price extremes.',
          methodology: 'Identifies overbought and oversold conditions using statistical measures and trades against the prevailing trend.',
          riskProfile: 'Medium - Can face extended periods of unrealized losses',
          bestMarkets: 'Range-bound markets, high-frequency trading, pairs trading',
          parameters: ['Lookback Period: 30 days', 'Z-Score Threshold: 2.0', 'Hold Period: 5-10 days'],
          pros: ['Profits from market inefficiencies', 'Good in sideways markets', 'Natural stop-loss levels'],
          cons: ['Poor in trending markets', 'Timing is critical', 'Can catch falling knives']
        };
      case 'breakout':
        return {
          overview: 'Breakout strategy identifies when price breaks through significant support or resistance levels, indicating potential new trends.',
          methodology: 'Monitors price ranges, volume patterns, and volatility to identify genuine breakouts versus false signals.',
          riskProfile: 'Medium to High - Early trend identification but prone to false breakouts',
          bestMarkets: 'Markets transitioning from consolidation to trending phases',
          parameters: ['Breakout Threshold: 2%', 'Volume Confirmation: 150%', 'Range Period: 20 days'],
          pros: ['Early trend identification', 'Clear entry signals', 'Strong risk-reward ratios'],
          cons: ['Many false breakouts', 'Requires quick execution', 'High transaction costs']
        };
      case 'volatility_spread':
        return {
          overview: 'Options strategy that profits from the difference between implied volatility (IV) and historical volatility (HV).',
          methodology: 'Compares current implied volatility to historical volatility to identify mispriced options and volatility opportunities.',
          riskProfile: 'Medium - Limited by option premium and time decay',
          bestMarkets: 'Options markets with high volatility skew, earnings seasons',
          parameters: ['IV-HV Threshold: 10%', 'Days to Expiration: 30-45', 'Delta Range: 0.3-0.7'],
          pros: ['Profits from volatility mispricing', 'Limited risk exposure', 'Market neutral potential'],
          cons: ['Requires options knowledge', 'Time decay risk', 'Liquidity constraints']
        };
      default:
        return {
          overview: 'Custom trading strategy with specific parameters and risk management rules.',
          methodology: 'Uses proprietary algorithms and indicators to generate trading signals.',
          riskProfile: 'Varies based on strategy configuration',
          bestMarkets: 'Depends on strategy design and parameters',
          parameters: ['Custom parameters based on strategy type'],
          pros: ['Tailored to specific market conditions', 'Customizable parameters'],
          cons: ['Requires backtesting', 'May need optimization']
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
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="methodology">Methodology</TabsTrigger>
            <TabsTrigger value="parameters">Parameters</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Info className="h-4 w-4" />
                  Strategy Overview
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-gray-600">{strategyInfo.overview}</p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-sm mb-2">Risk Profile</h4>
                    <Badge variant="outline">{strategyInfo.riskProfile}</Badge>
                  </div>
                  <div>
                    <h4 className="font-medium text-sm mb-2">Best Markets</h4>
                    <p className="text-sm text-gray-600">{strategyInfo.bestMarkets}</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-sm mb-2 text-green-600">Advantages</h4>
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
                    <h4 className="font-medium text-sm mb-2 text-red-600">Considerations</h4>
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
                <CardTitle>Strategy Methodology</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-600 leading-relaxed">{strategyInfo.methodology}</p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="parameters" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Strategy Parameters</CardTitle>
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
                <CardTitle>Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-gray-50 rounded">
                    <div className="text-lg font-bold text-gray-900">--</div>
                    <div className="text-xs text-gray-500">Total Return</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded">
                    <div className="text-lg font-bold text-gray-900">--</div>
                    <div className="text-xs text-gray-500">Sharpe Ratio</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded">
                    <div className="text-lg font-bold text-gray-900">--</div>
                    <div className="text-xs text-gray-500">Max Drawdown</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded">
                    <div className="text-lg font-bold text-gray-900">--</div>
                    <div className="text-xs text-gray-500">Win Rate</div>
                  </div>
                </div>
                <p className="text-sm text-gray-500 mt-4">
                  Performance metrics will be available after running a backtest.
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
    return <div>Loading strategies...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
        <div>
          <h2 className="text-lg font-semibold">Active Strategies</h2>
          <p className="text-sm text-muted-foreground">
            Manage and monitor your trading strategies
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <Button variant="outline" size="sm" asChild>
            <Link href="/options?tab=strategies" className="flex items-center gap-1">
              <Activity className="h-4 w-4" />
              Options Strategies
            </Link>
          </Button>
          <Button onClick={() => setShowNewStrategyModal(true)}>
            <Plus className="h-4 w-4 mr-2" />
            New Strategy
          </Button>
        </div>
      </div>

      {/* New Strategy Modal */}
      <Dialog open={showNewStrategyModal} onOpenChange={setShowNewStrategyModal}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Create New Trading Strategy</DialogTitle>
            <DialogDescription>
              Configure a new trading strategy with custom parameters and risk management settings.
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
              <h3 className="text-lg font-medium mb-2">No Strategies Found</h3>
              <p className="text-muted-foreground mb-4">
                Create your first trading strategy to get started
              </p>
              <Button onClick={() => setShowNewStrategyModal(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Create Strategy
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
                    {strategy.is_active ? 'Active' : 'Inactive'}
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
                    <span className="text-gray-400">Type:</span>
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
                      Backtest
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="flex-1 min-w-0"
                      onClick={() => handleViewDetails(strategy)}
                    >
                      View Details
                    </Button>
                    {strategy.strategy_type === 'volatility_spread' && (
                      <Button size="sm" variant="outline" className="w-full" asChild>
                        <Link href="/options?tab=strategies">
                          <ExternalLink className="h-3 w-3 mr-1 shrink-0" />
                          Open in Options
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