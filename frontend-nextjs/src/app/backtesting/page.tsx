import { redirect } from 'next/navigation';

/**
 * Backtesting is integrated into the Strategies page.
 * Redirect so /backtesting and deep links land on Strategies > Backtesting tab.
 */
export default function BacktestingPage() {
  redirect('/strategies?tab=backtesting');
}
