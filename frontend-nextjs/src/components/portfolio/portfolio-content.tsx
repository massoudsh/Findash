'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { formatCurrency, formatPercentage } from '@/lib/utils';
import { getPortfolios, getPositions } from '@/lib/services/api';
import { Plus } from 'lucide-react';

interface Portfolio {
  id: string;
  name: string;
  description: string;
  initial_capital: number;
  current_value: number;
  cash_balance: number;
  total_return: number;
  total_return_percent: number;
  risk_tolerance: string;
  created_at: string;
  updated_at: string;
}

interface Position {
  symbol: string;
  quantity: number;
  average_cost: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
  weight: number;
}

export function PortfolioContent() {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState<Portfolio | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchPortfolios() {
      try {
        const response = await getPortfolios();
        const portfolioData = response.data;
        setPortfolios(portfolioData);
        if (portfolioData.length > 0) {
          setSelectedPortfolio(portfolioData[0]);
        }
      } catch (error) {
        console.error('Error fetching portfolios:', error);
        // Set mock data on error
        const mockPortfolios = [
          {
            id: 'portfolio_1',
            name: 'Main Portfolio',
            description: 'Primary trading portfolio',
            initial_capital: 100000,
            current_value: 125000,
            cash_balance: 25000,
            total_return: 25000,
            total_return_percent: 25.0,
            risk_tolerance: 'moderate',
            created_at: '2024-01-01T00:00:00Z',
            updated_at: '2024-01-01T00:00:00Z'
          },
          {
            id: 'portfolio_2',
            name: 'Tech Stocks',
            description: 'Technology sector focused portfolio',
            initial_capital: 50000,
            current_value: 62000,
            cash_balance: 12000,
            total_return: 12000,
            total_return_percent: 24.0,
            risk_tolerance: 'aggressive',
            created_at: '2024-01-15T00:00:00Z',
            updated_at: '2024-01-15T00:00:00Z'
          }
        ];
        setPortfolios(mockPortfolios);
        setSelectedPortfolio(mockPortfolios[0]);
      } finally {
        setIsLoading(false);
      }
    }

    fetchPortfolios();
  }, []);

  useEffect(() => {
    if (selectedPortfolio) {
      async function fetchPositions() {
        try {
          const response = await getPositions(selectedPortfolio.id);
          setPositions(response.data);
        } catch (error) {
          console.error('Error fetching positions:', error);
          // Set mock positions on error
          setPositions([
            {
              symbol: 'AAPL',
              quantity: 100,
              average_cost: 150.00,
              market_value: 17500,
              unrealized_pnl: 2500,
              unrealized_pnl_percent: 16.67,
              weight: 52.2
            },
            {
              symbol: 'MSFT',
              quantity: 50,
              average_cost: 300.00,
              market_value: 16000,
              unrealized_pnl: 1000,
              unrealized_pnl_percent: 6.67,
              weight: 47.8
            }
          ]);
        }
      }

      fetchPositions();
    }
  }, [selectedPortfolio]);

  if (isLoading) {
    return <div>Loading portfolios...</div>;
  }

  return (
    <div className="space-y-6">
      {/* Portfolio Overview */}
      <div className="grid gap-4 md:grid-cols-3">
        {portfolios.map((portfolio) => (
          <Card 
            key={portfolio.id}
            className={`cursor-pointer transition-colors ${
              selectedPortfolio?.id === portfolio.id ? 'ring-2 ring-primary' : ''
            }`}
            onClick={() => setSelectedPortfolio(portfolio)}
          >
            <CardHeader>
              <CardTitle className="text-lg">{portfolio.name}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Current Value:</span>
                  <span className="font-medium">{formatCurrency(portfolio.current_value)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Initial Capital:</span>
                  <span className="font-medium">{formatCurrency(portfolio.initial_capital)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">P&L:</span>
                  <span className={`font-medium ${
                    portfolio.current_value - portfolio.initial_capital >= 0 
                      ? 'text-green-600' 
                      : 'text-red-600'
                  }`}>
                    {portfolio.current_value - portfolio.initial_capital >= 0 ? '+' : ''}
                    {formatCurrency(portfolio.current_value - portfolio.initial_capital)}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Selected Portfolio Positions */}
      {selectedPortfolio && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Positions - {selectedPortfolio.name}</CardTitle>
            <Button size="sm">
              <Plus className="h-4 w-4 mr-2" />
              Add Position
            </Button>
          </CardHeader>
          <CardContent>
            {positions.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No positions found for this portfolio
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2">Symbol</th>
                      <th className="text-right py-2">Quantity</th>
                      <th className="text-right py-2">Avg Price</th>
                      <th className="text-right py-2">Current Price</th>
                      <th className="text-right py-2">Market Value</th>
                      <th className="text-right py-2">P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((position, index) => (
                      <tr key={`${position.symbol}-${index}`} className="border-b">
                        <td className="py-2 font-medium">{position.symbol}</td>
                        <td className="text-right py-2">{position.quantity}</td>
                        <td className="text-right py-2">{formatCurrency(position.average_cost)}</td>
                        <td className="text-right py-2">{formatCurrency(position.market_value / position.quantity)}</td>
                        <td className="text-right py-2">{formatCurrency(position.market_value)}</td>
                        <td className={`text-right py-2 ${
                          position.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {position.unrealized_pnl >= 0 ? '+' : ''}
                          {formatCurrency(position.unrealized_pnl)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
} 