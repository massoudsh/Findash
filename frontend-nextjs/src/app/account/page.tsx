'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import dynamic from 'next/dynamic';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { User, Settings, CreditCard } from 'lucide-react';

const ProfilePage = dynamic(() => import('@/app/profile/page').then((m) => m.default), {
  ssr: false,
  loading: () => <div className="p-6 text-muted-foreground">در حال بارگذاری پروفایل…</div>,
});

const SettingsPage = dynamic(() => import('@/app/settings/page').then((m) => m.default), {
  ssr: false,
  loading: () => <div className="p-6 text-muted-foreground">در حال بارگذاری تنظیمات…</div>,
});

const SubscriptionPage = dynamic(() => import('@/app/account/subscription/page').then((m) => m.default), {
  ssr: false,
  loading: () => <div className="p-6 text-muted-foreground">در حال بارگذاری اشتراک…</div>,
});

type AccountTab = 'profile' | 'subscription' | 'settings';

export default function AccountPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const tabParam = searchParams.get('tab');
  const [activeTab, setActiveTab] = useState<AccountTab>(() => {
    if (tabParam === 'settings' || tabParam === 'profile' || tabParam === 'subscription') return tabParam;
    return 'profile';
  });

  useEffect(() => {
    if (tabParam === 'settings' || tabParam === 'profile' || tabParam === 'subscription') setActiveTab(tabParam);
  }, [tabParam]);

  function handleTabChange(value: string) {
    const tab = value as AccountTab;
    setActiveTab(tab);
    const params = new URLSearchParams(searchParams.toString());
    if (tab === 'profile') params.delete('tab');
    else params.set('tab', tab);
    const query = params.toString();
    router.replace(query ? `/account?${query}` : '/account', { scroll: false });
  }

  return (
    <div className="container mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">حساب کاربری</h1>
        <p className="text-muted-foreground">
          پروفایل و تنظیمات برنامه
        </p>
      </div>
      <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
        <TabsList className="grid w-full max-w-md grid-cols-3">
          <TabsTrigger value="profile" className="flex items-center gap-2">
            <User className="h-4 w-4" />
            پروفایل
          </TabsTrigger>
          <TabsTrigger value="subscription" className="flex items-center gap-2">
            <CreditCard className="h-4 w-4" />
            اشتراک
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            تنظیمات
          </TabsTrigger>
        </TabsList>
        <TabsContent value="profile" className="mt-6">
          <ProfilePage />
        </TabsContent>
        <TabsContent value="subscription" className="mt-6">
          <SubscriptionPage />
        </TabsContent>
        <TabsContent value="settings" className="mt-6">
          <SettingsPage />
        </TabsContent>
      </Tabs>
    </div>
  );
}
