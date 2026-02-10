'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { runPortfolioOptimization } from '@/lib/services/api';
import { formatPercentage } from '@/lib/utils';
import { AllocationChart } from '@/components/visualization/allocation-chart';

interface OptimizationResult {
  optimal_weights: Record<string, number>;
  expected_annual_return: number;
  expected_annual_volatility: number;
  sharpe_ratio: number;
}

export function PortfolioOptimizer() {
  const [symbols, setSymbols] = useState('BTC-USD,ETH-USD');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<OptimizationResult | null>(null);

  const handleOptimize = async () => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const symbolsArray = symbols.split(',').map(s => s.trim()).filter(s => s);
      if (symbolsArray.length < 2) {
        throw new Error("Please provide at least two symbols, separated by commas.");
      }
      
      const optimizationResult = await runPortfolioOptimization(symbolsArray, startDate, endDate);
      
      if (optimizationResult.status === 'success') {
        setResult(optimizationResult);
      } else {
        throw new Error(optimizationResult.message || 'Optimization failed');
      }

    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Portfolio Optimizer (Max Sharpe Ratio)</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid gap-4 md:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="symbols">Symbols (comma-separated)</Label>
            <Input 
              id="symbols" 
              value={symbols} 
              onChange={(e) => setSymbols(e.target.value)}
              placeholder="e.g., AAPL,GOOGL,MSFT"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="start-date">Start Date</Label>
            <Input 
              id="start-date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="end-date">End Date</Label>
            <Input
              id="end-date"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
        </div>
        
        <Button onClick={handleOptimize} disabled={isLoading}>
          {isLoading ? 'Optimizing...' : 'Run Optimization'}
        </Button>

        {error && (
          <div className="text-red-600 bg-red-100 p-3 rounded-md">
            <p className="font-bold">Error</p>
            <p>{error}</p>
          </div>
        )}

        {result && (
          <div className="space-y-4 pt-4">
            <h3 className="text-xl font-semibold">Optimization Results</h3>
            <div className="grid gap-4 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Key Metrics</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Exp. Annual Return:</span>
                    <span className="font-bold text-green-600">{formatPercentage(result.expected_annual_return)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Exp. Annual Volatility:</span>
                    <span className="font-bold">{formatPercentage(result.expected_annual_volatility)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Sharpe Ratio:</span>
                    <span className="font-bold">{result.sharpe_ratio.toFixed(2)}</span>
                  </div>
                </CardContent>
              </Card>
              <Card>
                 <CardHeader>
                  <CardTitle>Optimal Allocation</CardTitle>
                </CardHeader>
                <CardContent>
                  <AllocationChart 
                    data={Object.entries(result.optimal_weights).map(([name, value]) => ({ name, value }))} 
                  />
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 