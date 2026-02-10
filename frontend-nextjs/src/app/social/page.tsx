import { Suspense } from 'react';
import { SocialContent } from '@/components/social/social-content';

export default function SocialPage() {
  return (
    <div className="container mx-auto px-6 py-8">
      <Suspense fallback={<div className="text-center">Loading social signals...</div>}>
        <SocialContent />
      </Suspense>
    </div>
  );
}
