'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { formatCurrency, formatDate } from '@/lib/utils';
import { getTrades, Trade } from '@/lib/services/trades_api';
import { getPortfolios, Portfolio } from '@/lib/services/portfolio_api'; // Assuming portfolio API exists
import { Plus } from 'lucide-react';

export function TradesContent() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState<number | 'all'>('all');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      setIsLoading(true);
      try {
        const [portfolioData, tradesData] = await Promise.all([
            getPortfolios(),
            getTrades()
        ]);
        
        setPortfolios(portfolioData);
        const sortedTrades = tradesData.sort((a, b) => new Date(b.executed_at).getTime() - new Date(a.executed_at).getTime());
        setTrades(sortedTrades);

      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchData();
  }, []);

  const filteredTrades = selectedPortfolio === 'all' 
    ? trades 
    : trades.filter(trade => trade.portfolio_id === selectedPortfolio);

  if (isLoading) {
    return <div>در حال بارگذاری معاملات...</div>;
  }

  return (
    <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Trade History</CardTitle>
          <div className="flex gap-2">
            <select
                value={selectedPortfolio}
                onChange={(e) => setSelectedPortfolio(e.target.value === 'all' ? 'all' : Number(e.target.value))}
                className="px-3 py-2 border rounded-md bg-background"
            >
                <option value="all">All Portfolios</option>
                {portfolios.map((portfolio) => (
                <option key={portfolio.id} value={portfolio.id}>
                    {portfolio.name}
                </option>
                ))}
            </select>
            <Button variant="outline">
                <Plus className="h-4 w-4 mr-2" />
                New Trade
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {filteredTrades.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No trades found
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2">Date</th>
                    <th className="text-left py-2">Symbol</th>
                    <th className="text-left py-2">Side</th>
                    <th className="text-right py-2">Quantity</th>
                    <th className="text-right py-2">Price</th>
                    <th className="text-right py-2">Total</th>
                    <th className="text-left py-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTrades.map((trade) => (
                    <tr key={trade.id} className="border-b">
                      <td className="py-2 text-sm">{formatDate(trade.executed_at)}</td>
                      <td className="py-2 font-medium">{trade.symbol}</td>
                      <td className="py-2">
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          trade.side === 'buy' 
                            ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                            : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                        }`}>
                          {trade.side.toUpperCase()}
                        </span>
                      </td>
                      <td className="text-right py-2">{trade.quantity}</td>
                      <td className="text-right py-2">{formatCurrency(trade.price)}</td>
                      <td className="text-right py-2">{formatCurrency(trade.price * trade.quantity)}</td>
                      <td className="py-2">
                        <span className={`px-2 py-1 rounded-full text-xs capitalize ${
                          trade.status === 'executed' 
                            ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                            : trade.status === 'pending'
                            ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300'
                            : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
                        }`}>
                          {trade.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
  );
} 