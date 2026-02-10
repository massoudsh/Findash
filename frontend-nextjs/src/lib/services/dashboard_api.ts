export interface PortfolioPerformance {
  date: string;
  value: number;
}

const mockPerformance: PortfolioPerformance[] = [
  { date: '2023-10-01', value: 100000 },
  { date: '2023-10-02', value: 101500 },
  { date: '2023-10-03', value: 101000 },
  { date: '2023-10-04', value: 103200 },
  { date: '2023-10-05', value: 105500 },
  { date: '2023-10-06', value: 104800 },
  { date: '2023-10-09', value: 106200 },
  { date: '2023-10-10', value: 108000 },
  { date: '2023-10-11', value: 107500 },
  { date: '2023-10-12', value: 109300 },
  { date: '2023-10-13', value: 110500 },
  { date: '2023-10-16', value: 112000 },
  { date: '2023-10-17', value: 113500 },
  { date: '2023-10-18', value: 113000 },
  { date: '2023-10-19', value: 115200 },
  { date: '2023-10-20', value: 117500 },
  { date: '2023-10-23', value: 116800 },
  { date: '2023-10-24', value: 118200 },
  { date: '2023-10-25', value: 120000 },
  { date: '2023-10-26', value: 122500 },
  { date: '2023-10-27', value: 125000 },
];

export async function getPortfolioPerformance(): Promise<PortfolioPerformance[]> {
  console.log("Mock API: Fetching portfolio performance data.");
  return new Promise(resolve => {
    setTimeout(() => resolve(mockPerformance), 500);
  });
} 