import type { Metadata } from 'next';
import './globals.css';
import { NavigationWrapper } from '@/components/navigation/navigation-wrapper';
import { SessionProviderWrapper } from '@/components/auth/session-provider-wrapper';
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
    <html lang="en" className="dark">
      <head>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
      </head>
      <body className="font-sans">
        <ErrorBoundary>
          <ChunkErrorHandler />
          <SessionProviderWrapper>
            <NavigationWrapper>
              {children}
            </NavigationWrapper>
            <Toaster />
          </SessionProviderWrapper>
        </ErrorBoundary>
      </body>
    </html>
  );
} 