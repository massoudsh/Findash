export interface Portfolio {
  id: number;
  name: string;
  description: string;
  initial_capital: number;
  current_value: number;
  created_at: string;
}

export interface Position {
  id: number;
  symbol: string;
  quantity: number;
  average_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
}

const mockPortfolios: Portfolio[] = [
  { id: 1, name: 'Growth Stocks', description: 'High-risk, high-reward tech stocks.', initial_capital: 100000, current_value: 125000, created_at: '2022-01-15T09:30:00Z' },
  { id: 2, name: 'Dividend Income', description: 'Stable, income-generating assets.', initial_capital: 250000, current_value: 265000, created_at: '2021-06-20T14:00:00Z' },
];

const mockPositions: { [key: number]: Position[] } = {
  1: [
    { id: 101, symbol: 'AAPL', quantity: 50, average_price: 150, current_price: 175, market_value: 8750, unrealized_pnl: 1250 },
    { id: 102, symbol: 'GOOGL', quantity: 20, average_price: 2800, current_price: 2950, market_value: 59000, unrealized_pnl: 3000 },
    { id: 103, symbol: 'TSLA', quantity: 15, average_price: 800, current_price: 750, market_value: 11250, unrealized_pnl: -750 },
  ],
  2: [
    { id: 201, symbol: 'JNJ', quantity: 100, average_price: 160, current_price: 170, market_value: 17000, unrealized_pnl: 1000 },
    { id: 202, symbol: 'PG', quantity: 150, average_price: 140, current_price: 145, market_value: 21750, unrealized_pnl: 750 },
    { id: 203, symbol: 'KO', quantity: 200, average_price: 55, current_price: 60, market_value: 12000, unrealized_pnl: 1000 },
  ]
};

export async function getPortfolios(): Promise<Portfolio[]> {
  console.log("Mock API: Fetching portfolios data.");
  return new Promise(resolve => {
    setTimeout(() => resolve(mockPortfolios), 300);
  });
}

export async function getPositions(portfolioId: number): Promise<Position[]> {
  console.log(`Mock API: Fetching positions for portfolio ${portfolioId}.`);
  return new Promise(resolve => {
    setTimeout(() => resolve(mockPositions[portfolioId] || []), 400);
  });
} 