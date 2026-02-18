import { redirect } from 'next/navigation';

/**
 * Risk management lives in the Command Center (Trading page).
 */
export default function RiskPage() {
  redirect('/trading?tab=risk');
}
