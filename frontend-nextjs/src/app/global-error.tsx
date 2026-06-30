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
                {isChunkLoadError ? 'خطا در بارگذاری' : 'خطای برنامه'}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-sm text-muted-foreground">
                {isChunkLoadError ? (
                  <p>
                    برنامه نتوانست منابع لازم را بارگذاری کند. این مشکل معمولاً پس از به‌روزرسانی برنامه رخ می‌دهد.
                    لطفاً صفحه را رفرش کنید تا نسخه جدید بارگذاری شود.
                  </p>
                ) : (
                  <p>
                    خطای بحرانی در برنامه رخ داده است. لطفاً صفحه را رفرش کنید
                    یا در صورت ادامه مشکل با پشتیبانی تماس بگیرید.
                  </p>
                )}
              </div>

              {process.env.NODE_ENV === 'development' && (
                <details className="text-xs bg-muted p-2 rounded">
                  <summary className="cursor-pointer font-medium mb-2">
                    جزئیات خطا
                  </summary>
                  <pre className="whitespace-pre-wrap break-all">
                    {error.message}
                    {error.stack}
                  </pre>
                </details>
              )}

              <div className="flex gap-2">
                <Button onClick={handleReload} variant="default" size="sm">
                  <RefreshCw className="h-4 w-4 ms-2" />
                  رفرش صفحه
                </Button>
                {!isChunkLoadError && (
                  <Button onClick={reset} variant="outline" size="sm">
                    تلاش مجدد
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