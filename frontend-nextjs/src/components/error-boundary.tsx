'use client';

import { Component, ErrorInfo, ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  isRetrying: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  private retryCount = 0;
  private maxRetries = 3;

  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      isRetrying: false,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
      isRetrying: false,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo,
    });

    // Auto-retry for ChunkLoadError
    if (this.isChunkLoadError(error) && this.retryCount < this.maxRetries) {
      this.handleRetry();
    }
  }

  private isChunkLoadError(error: Error): boolean {
    return (
      error.name === 'ChunkLoadError' ||
      error.message.includes('Loading chunk') ||
      error.message.includes('Loading CSS chunk')
    );
  }

  private handleRetry = async () => {
    this.retryCount++;
    this.setState({ isRetrying: true });

    try {
      // Wait a bit before retrying
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // For chunk load errors, try to reload the page
      if (this.state.error && this.isChunkLoadError(this.state.error)) {
        window.location.reload();
        return;
      }

      // Reset the error boundary
      this.setState({
        hasError: false,
        error: null,
        errorInfo: null,
        isRetrying: false,
      });
    } catch (retryError) {
      console.error('Retry failed:', retryError);
      this.setState({ isRetrying: false });
    }
  };

  private handleManualRetry = () => {
    this.retryCount = 0;
    this.handleRetry();
  };

  private handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const isChunkError = this.state.error && this.isChunkLoadError(this.state.error);

      return (
        <div className="min-h-screen flex items-center justify-center p-4">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-destructive">
                <AlertTriangle className="h-5 w-5" />
                {isChunkError ? 'Loading Error' : 'Something went wrong'}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-sm text-muted-foreground">
                {isChunkError ? (
                  <p>
                    The application failed to load some resources. This usually happens
                    when the app has been updated. Please try refreshing the page.
                  </p>
                ) : (
                  <p>
                    An unexpected error occurred. Please try refreshing the page or
                    contact support if the problem persists.
                  </p>
                )}
              </div>

              {process.env.NODE_ENV === 'development' && this.state.error && (
                <details className="text-xs bg-muted p-2 rounded">
                  <summary className="cursor-pointer font-medium mb-2">
                    Error Details
                  </summary>
                  <pre className="whitespace-pre-wrap break-all">
                    {this.state.error.toString()}
                    {this.state.errorInfo?.componentStack}
                  </pre>
                </details>
              )}

              <div className="flex gap-2">
                <Button
                  onClick={this.handleReload}
                  variant="default"
                  size="sm"
                  disabled={this.state.isRetrying}
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh Page
                </Button>
                
                {!isChunkError && (
                  <Button
                    onClick={this.handleManualRetry}
                    variant="outline"
                    size="sm"
                    disabled={this.state.isRetrying}
                  >
                    {this.state.isRetrying ? 'Retrying...' : 'Try Again'}
                  </Button>
                )}
              </div>

              {this.retryCount > 0 && (
                <p className="text-xs text-muted-foreground">
                  Retry attempts: {this.retryCount}/{this.maxRetries}
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
} 