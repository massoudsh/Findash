import { getServerSession } from 'next-auth';
import { redirect } from 'next/navigation';
import { authOptions } from '@/lib/auth-options';

export default async function AdminLayout({ children }: { children: React.ReactNode }) {
  const session = await getServerSession(authOptions);
  if (!session) redirect('/auth/signin?callbackUrl=/admin');
  if (session.user.role !== 'admin') {
    return <p role="alert" className="p-6">دسترسی محدود به مدیران سیستم است.</p>;
  }
  return children;
}
