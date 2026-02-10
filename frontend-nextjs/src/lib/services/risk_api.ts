export interface RiskMetrics {
  portfolioId: string;
  var_95: number;
  sharpe_ratio: number;
  max_drawdown: number;
  leverage: number;
}

export interface StressTestResult {
  scenario: string;
  impact: number;
  pnl: number;
}

const mockMetrics: RiskMetrics = {
  portfolioId: 'Growth Stocks',
  var_95: 12500.75,
  sharpe_ratio: 1.8,
  max_drawdown: 0.15,
  leverage: 1.2,
};

const mockStressTests: StressTestResult[] = [
  { scenario: '2008 Financial Crisis', impact: -0.35, pnl: -43750 },
  { scenario: 'COVID-19 Crash', impact: -0.25, pnl: -31250 },
  { scenario: 'Interest Rate Hike (50bps)', impact: -0.05, pnl: -6250 },
];

export async function getRiskMetrics(): Promise<RiskMetrics> {
  console.log("Mock API: Fetching risk metrics.");
  return new Promise(resolve => {
    setTimeout(() => resolve(mockMetrics), 700);
  });
}

export async function getStressTestResults(): Promise<StressTestResult[]> {
  console.log("Mock API: Fetching stress test results.");
  return new Promise(resolve => {
    setTimeout(() => resolve(mockStressTests), 900);
  });
} 