'use client';

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Terminal, CheckCircle, Clock } from 'lucide-react';

// --- Type Definitions ---

interface BacktestMetrics {
  total_return: number;
  annual_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  winning_days: number;
  backtest_length_days: number;
}

interface BacktestResult {
  equity_curve: number[];
  drawdown: number[];
  timestamps: string[];
  strategy_name: string;
  metrics: BacktestMetrics;
}

interface StrategyParams {
  short_window: number;
  long_window: number;
  max_position_size: number;
  use_geo_risk: boolean;
  use_prophet: boolean;
  name: string;
}

interface BacktestRequest {
  tickers: string[];
  start_date: string;
  end_date: string;
  capital_base: number;
  strategy_params: StrategyParams;
}

// --- Helper Components ---

function MetricCard({ title, value }: { title: string; value: string }) {
  return (
    <Card className="text-center">
      <CardHeader className="p-4">
        <CardDescription>{title}</CardDescription>
      </CardHeader>
      <CardContent className="p-4">
        <p className="text-2xl font-bold">{value}</p>
      </CardContent>
    </Card>
  );
}

// --- Main Component ---

export function StrategyBacktestResults() {
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Default form state matching the API request structure
  const [formState, setFormState] = useState<BacktestRequest>({
    tickers: ['AAPL', 'MSFT', 'GOOGL'],
    start_date: '2020-01-01',
    end_date: '2022-01-01',
    capital_base: 100000,
    strategy_params: {
      name: 'Quantum Crossover',
      short_window: 20,
      long_window: 50,
      max_position_size: 0.2,
      use_geo_risk: true,
      use_prophet: true,
    },
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type } = e.target;
    
    if (name in formState.strategy_params) {
        setFormState(prev => ({
            ...prev,
            strategy_params: { ...prev.strategy_params, [name]: type === 'number' ? parseFloat(value) : value }
        }));
    } else if (name === 'tickers') {
        setFormState(prev => ({ ...prev, tickers: value.split(',').map(t => t.trim()) }));
    } else {
        setFormState(prev => ({ ...prev, [name]: type === 'number' ? parseFloat(value) : value }));
    }
  };

  const stopPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };

  const checkTaskStatus = useCallback(async (currentTaskId: string) => {
    try {
      const res = await fetch(`/api/strategies/results/${currentTaskId}`);
      if (!res.ok) throw new Error('Failed to check task status.');
      
      const data = await res.json();
      
      if (data.status === 'SUCCESS') {
        stopPolling();
        setResult(data.result);
        setIsLoading(false);
        setTaskId(null);
      } else if (data.status === 'FAILURE') {
        stopPolling();
        setError(data.result?.details || 'Backtest task failed. Check server logs.');
        setIsLoading(false);
        setTaskId(null);
      }
      // If status is PENDING or STARTED, do nothing and let the polling continue.
    } catch (e: any) {
      stopPolling();
      setError(e.message);
      setIsLoading(false);
      setTaskId(null);
    }
  }, []);

  useEffect(() => {
    if (taskId) {
      pollingIntervalRef.current = setInterval(() => {
        checkTaskStatus(taskId);
      }, 3000); // Poll every 3 seconds
    }
    return stopPolling; // Cleanup on component unmount or if taskId changes
  }, [taskId, checkTaskStatus]);

  const submitBacktest = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('/api/strategies/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formState),
      });
      
      if (res.status !== 202) {
        const errorData = await res.json();
        throw new Error(errorData.detail || `HTTP ${res.status}: Failed to submit task`);
      }
      
      const { task_id } = await res.json();
      setTaskId(task_id); // This will trigger the useEffect to start polling

    } catch (e: any) {
      setError(e.message || 'An unknown error occurred.');
      setIsLoading(false);
    }
  }, [formState]);

  const chartData = result?.timestamps.map((ts, i) => ({
    name: new Date(ts).toLocaleDateString(),
    equity: result.equity_curve[i],
    drawdown: result.drawdown[i] * 100, // as percentage
  }));
  
  const metrics = result?.metrics;

  return (
    <div className="space-y-6 p-4 md:p-6">
      <Card>
        <CardHeader>
          <CardTitle>Run a New Backtest</CardTitle>
          <CardDescription>Configure and execute a backtest using the Quantum Engine.</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="space-y-2">
            <Label htmlFor="tickers">Tickers (comma-separated)</Label>
            <Input id="tickers" name="tickers" value={formState.tickers.join(', ')} onChange={handleInputChange} />
          </div>
          <div className="space-y-2">
            <Label htmlFor="start_date">Start Date</Label>
            <Input id="start_date" name="start_date" type="date" value={formState.start_date} onChange={handleInputChange} />
          </div>
          <div className="space-y-2">
            <Label htmlFor="end_date">End Date</Label>
            <Input id="end_date" name="end_date" type="date" value={formState.end_date} onChange={handleInputChange} />
          </div>
          <div className="space-y-2">
            <Label htmlFor="capital_base">Capital Base</Label>
            <Input id="capital_base" name="capital_base" type="number" value={formState.capital_base} onChange={handleInputChange} />
          </div>
        </CardContent>
        <CardFooter>
          <Button onClick={submitBacktest} disabled={isLoading}>
            {isLoading ? 'Running...' : 'Run Backtest'}
          </Button>
        </CardFooter>
      </Card>
      
      {isLoading && (
        <Alert>
            <Clock className="h-4 w-4" />
            <AlertTitle>Backtest in Progress</AlertTitle>
            <AlertDescription>Your backtest is running in the background. Results will appear here when complete. (Task ID: {taskId})</AlertDescription>
        </Alert>
      )}

      {error && (
         <Alert variant="destructive">
            <Terminal className="h-4 w-4" />
            <AlertTitle>Backtest Failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
         </Alert>
      )}

      {result && metrics && chartData && (
        <div className="space-y-6 animate-in fade-in-50">
          <Alert variant="default" className="bg-green-50 border-green-200">
             <CheckCircle className="h-4 w-4 text-green-600" />
             <AlertTitle className="text-green-800">Backtest Complete</AlertTitle>
             <AlertDescription className="text-green-700">The results for your backtest are ready.</AlertDescription>
          </Alert>

          <Card>
            <CardHeader>
              <CardTitle>Backtest Results: {result.strategy_name}</CardTitle>
              <CardDescription>
                Ran on {metrics.backtest_length_days} periods from {formState.start_date} to {formState.end_date}.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 md:grid-cols-3 gap-4">
               <MetricCard title="Total Return" value={`${(metrics.total_return * 100).toFixed(2)}%`} />
               <MetricCard title="Annual Return" value={`${(metrics.annual_return * 100).toFixed(2)}%`} />
               <MetricCard title="Sharpe Ratio" value={metrics.sharpe_ratio.toFixed(2)} />
               <MetricCard title="Max Drawdown" value={`${(metrics.max_drawdown * 100).toFixed(2)}%`} />
               <MetricCard title="Win Rate" value={`${(metrics.winning_days * 100).toFixed(2)}%`} />
               <MetricCard title="Periods" value={metrics.backtest_length_days.toString()} />
            </CardContent>
          </Card>

          <Card>
             <CardHeader>
                <CardTitle>Equity Curve</CardTitle>
             </CardHeader>
             <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis domain={['auto', 'auto']} />
                        <Tooltip formatter={(value: number) => `$${value.toLocaleString()}`} />
                        <Area type="monotone" dataKey="equity" stroke="#2563eb" fill="#bfdbfe" />
                    </AreaChart>
                </ResponsiveContainer>
             </CardContent>
          </Card>
          
          <Card>
             <CardHeader>
                <CardTitle>Drawdown</CardTitle>
             </CardHeader>
             <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                     <AreaChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis domain={['auto', 0]} />
                        <Tooltip formatter={(value: number) => `${value.toFixed(2)}%`} />
                        <Area type="monotone" dataKey="drawdown" stroke="#dc2626" fill="#fecaca" />
                    </AreaChart>
                </ResponsiveContainer>
             </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
} 