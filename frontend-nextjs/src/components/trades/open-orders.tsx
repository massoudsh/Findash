'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Trade, getTrades } from '@/lib/services/trades_api';
import { formatCurrency } from '@/lib/utils';
import { Button } from '@/components/ui/button';

export function OpenOrders() {
  const [openOrders, setOpenOrders] = useState<Trade[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchOpenOrders = async () => {
      setIsLoading(true);
      try {
        const allTrades = await getTrades();
        const pending = allTrades.filter(t => t.status === 'pending');
        setOpenOrders(pending);
      } catch (error) {
        console.error("Failed to fetch open orders:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchOpenOrders();
    // Refresh every 10 seconds
    const interval = setInterval(fetchOpenOrders, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Open Orders</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && openOrders.length === 0 ? (
          <p>Loading open orders...</p>
        ) : openOrders.length === 0 ? (
          <p className="text-muted-foreground">No open orders.</p>
        ) : (
          <div className="space-y-4">
            {openOrders.map(order => (
              <div key={order.id} className="flex items-center justify-between">
                <div>
                  <p className="font-semibold">{order.symbol} <span className={`text-sm ${order.side === 'buy' ? 'text-green-600' : 'text-red-600'}`}>{order.side.toUpperCase()}</span></p>
                  <p className="text-sm text-muted-foreground">
                    {order.quantity} @ {formatCurrency(order.price)}
                  </p>
                </div>
                <Button variant="outline" size="sm">Cancel</Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
} 