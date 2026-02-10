'use client';

import { useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error('Global error:', error);
  }, [error]);

  const isChunkLoadError = 
    error.name === 'ChunkLoadError' ||
    error.message.includes('Loading chunk') ||
    error.message.includes('Loading CSS chunk');

  const handleReload = () => {
    window.location.reload();
  };

  return (
    <html>
      <body>
        <div className="min-h-screen flex items-center justify-center p-4 bg-background">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-destructive">
                <AlertTriangle className="h-5 w-5" />
                {isChunkLoadError ? 'Loading Error' : 'Application Error'}
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
                    A critical error occurred in the application. Please refresh the page
                    or contact support if the problem persists.
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
                <Button onClick={handleReload} variant="default" size="sm">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh Page
                </Button>
                {!isChunkLoadError && (
                  <Button onClick={reset} variant="outline" size="sm">
                    Try Again
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </body>
    </html>
  );
} 