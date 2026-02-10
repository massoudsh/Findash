'use client';

import { useToast, Toast } from '@/components/ui/toast';

export function Toaster() {
  const { toasts } = useToast();

  return (
    <>
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className="fixed top-4 right-4 z-50 w-full max-w-sm"
        >
          <Toast {...toast} />
        </div>
      ))}
    </>
  );
} 