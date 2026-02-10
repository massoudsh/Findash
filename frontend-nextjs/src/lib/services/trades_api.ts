export interface Trade {
  id: number;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  executed_at: string;
  status: 'executed' | 'pending' | 'failed';
  portfolio_id: number;
}

const mockTrades: Trade[] = [
    { id: 1, symbol: 'AAPL', side: 'buy', quantity: 10, price: 175.2, executed_at: '2023-10-27T14:30:00Z', status: 'executed', portfolio_id: 1 },
    { id: 2, symbol: 'GOOGL', side: 'sell', quantity: 5, price: 2950, executed_at: '2023-10-27T12:15:00Z', status: 'executed', portfolio_id: 1 },
    { id: 3, symbol: 'TSLA', side: 'buy', quantity: 20, price: 740, executed_at: '2023-10-26T10:05:00Z', status: 'executed', portfolio_id: 2 },
    { id: 4, symbol: 'NVDA', side: 'buy', quantity: 15, price: 450, executed_at: '2023-10-28T09:00:00Z', status: 'pending', portfolio_id: 1 },
];

export async function getTrades(portfolioId?: number): Promise<Trade[]> {
  console.log(`Mock API: Fetching trades for portfolio ${portfolioId || 'all'}.`);
  return new Promise(resolve => {
    setTimeout(() => {
        if (portfolioId) {
            resolve(mockTrades.filter(t => t.portfolio_id === portfolioId));
        } else {
            resolve(mockTrades);
        }
    }, 600);
  });
}

export async function placeTrade(trade: Omit<Trade, 'id' | 'executed_at' | 'status'>): Promise<Trade> {
    console.log(`Mock API: Placing trade for ${trade.symbol}.`);
    const newTrade: Trade = {
        id: Math.max(...mockTrades.map(t => t.id)) + 1,
        ...trade,
        executed_at: new Date().toISOString(),
        status: 'pending',
    };
    mockTrades.unshift(newTrade);
    return new Promise(resolve => {
        setTimeout(() => resolve(newTrade), 300);
    });
} 