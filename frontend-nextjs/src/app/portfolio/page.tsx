import { redirect } from 'next/navigation';

/**
 * Portfolio lives inside the Command center (Dashboard).
 * Redirect so all "Portfolio" entry points land on the same view.
 */
export default function PortfolioPage() {
  redirect('/dashboard?tab=portfolio');
}
