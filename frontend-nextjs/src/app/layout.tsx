import type { Metadata } from 'next';
import { Suspense } from 'react';
import { Vazirmatn } from 'next/font/google';
import './globals.css';
import { NavigationWrapper } from '@/components/navigation/navigation-wrapper';
import { SessionProviderWrapper } from '@/components/auth/session-provider-wrapper';
import { LocaleProvider } from '@/lib/i18n/locale-context';
import { Toaster } from '@/components/ui/toaster';
import { ErrorBoundary } from '@/components/error-boundary';
import { ChunkErrorHandler } from '@/components/chunk-error-handler';

const vazir = Vazirmatn({
  subsets: ['arabic'],
  variable: '--font-vazir',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'فین‌دَش | پلتفرم هوشمند معاملاتی',
  description: 'پلتفرم هوشمند معاملاتی بازار ایران با هوش مصنوعی، تحلیل ریل‌تایم و مدیریت ریسک پیشرفته',
  icons: {
    icon: '/favicon.svg',
    apple: '/apple-touch-icon.png',
  },
  openGraph: {
    title: 'پلتفرم معاملاتی اکتپوس',
    description: 'پلتفرم هوشمند معاملاتی با هوش مصنوعی',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="fa" dir="rtl" className={`dark ${vazir.variable}`} suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
      </head>
      <body className="font-iran-yekan">
        <ErrorBoundary>
          <ChunkErrorHandler />
          <LocaleProvider>
            <SessionProviderWrapper>
              <Suspense fallback={<div className="flex min-h-screen items-center justify-center">در حال بارگذاری…</div>}>
                <NavigationWrapper>
                  {children}
                </NavigationWrapper>
              </Suspense>
              <Toaster />
            </SessionProviderWrapper>
          </LocaleProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
} 