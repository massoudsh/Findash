'use client';

import { useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error('Page error:', error);
  }, [error]);

  const isChunkLoadError = 
    error.name === 'ChunkLoadError' ||
    error.message.includes('Loading chunk') ||
    error.message.includes('Loading CSS chunk');

  const handleReload = () => {
    window.location.reload();
  };

  const handleGoHome = () => {
    window.location.href = '/dashboard';
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="h-5 w-5" />
            {isChunkLoadError ? 'Loading Error' : 'Something went wrong'}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="text-sm text-muted-foreground">
            {isChunkLoadError ? (
              <p>
                The application failed to load some resources. This usually happens
                when the app has been updated. Please refresh the page to get the
                latest version.
              </p>
            ) : (
              <p>
                An unexpected error occurred while loading this page. Please try
                refreshing or go back to the home page.
              </p>
            )}
          </div>

          {process.env.NODE_ENV === 'development' && (
            <details className="text-xs bg-muted p-2 rounded">
              <summary className="cursor-pointer font-medium mb-2">
                Error Details
              </summary>
              <pre className="whitespace-pre-wrap break-all">
                {error.message}
                {error.stack}
              </pre>
            </details>
          )}

          <div className="flex gap-2">
            {isChunkLoadError ? (
              <Button onClick={handleReload} variant="default" size="sm">
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh Page
              </Button>
            ) : (
              <>
                <Button onClick={reset} variant="default" size="sm">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Try Again
                </Button>
                <Button onClick={handleGoHome} variant="outline" size="sm">
                  <Home className="h-4 w-4 mr-2" />
                  Go Home
                </Button>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 