'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { placeTrade } from '@/lib/services/trades_api';

export function OrderEntry() {
  const [symbol, setSymbol] = useState('');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      await placeTrade({
        symbol,
        quantity: Number(quantity),
        price: Number(price),
        side,
        portfolio_id: 1, // Assuming a default portfolio for now
      });
      // Reset form
      setSymbol('');
      setQuantity('');
      setPrice('');
    } catch (error) {
      console.error("Failed to place trade:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Order Entry</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1">
            <Label htmlFor="symbol">Symbol</Label>
            <Input id="symbol" value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} placeholder="e.g., AAPL" required />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <Label htmlFor="quantity">Quantity</Label>
              <Input id="quantity" type="number" value={quantity} onChange={e => setQuantity(e.target.value)} placeholder="0" required />
            </div>
            <div className="space-y-1">
              <Label htmlFor="price">Price</Label>
              <Input id="price" type="number" value={price} onChange={e => setPrice(e.target.value)} placeholder="0.00" required />
            </div>
          </div>
          <div className="space-y-1">
            <Label>Side</Label>
            <select
              value={side}
              onChange={e => setSide(e.target.value as 'buy' | 'sell')}
              className="w-full px-3 py-2 border rounded-md bg-background"
            >
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </select>
          </div>
          <Button type="submit" disabled={isSubmitting} className="w-full">
            {isSubmitting ? 'Placing Order...' : 'Place Order'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
} 