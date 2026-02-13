import type { Metadata } from 'next';
import './globals.css';
import { NavigationWrapper } from '@/components/navigation/navigation-wrapper';
import { SessionProviderWrapper } from '@/components/auth/session-provider-wrapper';
import { LocaleProvider } from '@/lib/i18n/locale-context';
import { Toaster } from '@/components/ui/toaster';
import { ErrorBoundary } from '@/components/error-boundary';
import { ChunkErrorHandler } from '@/components/chunk-error-handler';

export const metadata: Metadata = {
  title: 'Octopus Trading Platform',
  description: 'Octopus - Intelligent Multi-Agent Trading Platform with AI-Powered Analytics, Deep Learning Models, and Quantum-Enhanced Investment Strategies',
  icons: {
    icon: '/favicon.svg',
    apple: '/apple-touch-icon.png',
  },
  openGraph: {
    title: 'Octopus Trading Platform',
    description: 'Intelligent Multi-Agent Trading Platform with AI-Powered Analytics',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700&display=swap" rel="stylesheet" />
      </head>
      <body className="font-sans">
        <ErrorBoundary>
          <ChunkErrorHandler />
          <LocaleProvider>
            <SessionProviderWrapper>
              <NavigationWrapper>
                {children}
              </NavigationWrapper>
              <Toaster />
            </SessionProviderWrapper>
          </LocaleProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
} 