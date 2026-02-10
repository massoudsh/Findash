'use client';

import { useEffect } from 'react';

export function ChunkErrorHandler() {
  useEffect(() => {
    // Handle unhandled promise rejections (like ChunkLoadError)
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      const error = event.reason;
      
      // Check if it's a ChunkLoadError
      if (
        error &&
        (error.name === 'ChunkLoadError' ||
         (typeof error === 'string' && error.includes('Loading chunk')) ||
         (error.message && error.message.includes('Loading chunk')))
      ) {
        console.warn('ChunkLoadError detected, reloading page...', error);
        
        // Prevent the default unhandled rejection behavior
        event.preventDefault();
        
        // Reload the page after a short delay
        setTimeout(() => {
          window.location.reload();
        }, 1000);
      }
    };

    // Handle global JavaScript errors
    const handleError = (event: ErrorEvent) => {
      const error = event.error;
      
      if (
        error &&
        (error.name === 'ChunkLoadError' ||
         (error.message && error.message.includes('Loading chunk')))
      ) {
        console.warn('ChunkLoadError detected in global error handler, reloading page...', error);
        
        // Prevent the default error behavior
        event.preventDefault();
        
        // Reload the page after a short delay
        setTimeout(() => {
          window.location.reload();
        }, 1000);
      }
    };

    // Add event listeners
    window.addEventListener('unhandledrejection', handleUnhandledRejection);
    window.addEventListener('error', handleError);

    // Cleanup
    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
      window.removeEventListener('error', handleError);
    };
  }, []);

  return null; // This component doesn't render anything
} 