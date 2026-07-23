'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Settings,
  Layout,
  Bell,
  Palette,
  Save,
  RefreshCw,
  RotateCcw,
  AlertCircle
} from 'lucide-react';

export function UserPreferences() {
  const [hasChanges, setHasChanges] = useState(false);
  const [saving, setSaving] = useState(false);

  const savePreferences = async () => {
    setSaving(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    setHasChanges(false);
    setSaving(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">تنظیمات کاربری</h2>
          <p className="text-muted-foreground">تجربه معاملاتی خود را شخصی‌سازی کنید</p>
        </div>

        <div className="flex items-center gap-2">
          {hasChanges && (
            <Badge variant="secondary" className="bg-yellow-100 text-yellow-800">
              <AlertCircle className="h-3 w-3 mr-1" />
              تغییرات ذخیره‌نشده
            </Badge>
          )}
          <Button variant="outline">
            <RotateCcw className="h-4 w-4 mr-2" />
            بازنشانی
          </Button>
          <Button onClick={savePreferences} disabled={saving || !hasChanges}>
            {saving ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            {saving ? 'در حال ذخیره...' : 'ذخیره تغییرات'}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="dashboard">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="dashboard">داشبورد</TabsTrigger>
          <TabsTrigger value="notifications">اعلان‌ها</TabsTrigger>
          <TabsTrigger value="display">نمایش</TabsTrigger>
          <TabsTrigger value="trading">معاملات</TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layout className="h-5 w-5" />
                ویجت‌های داشبورد
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                انتخاب کنید کدام ویجت‌ها در داشبورد شما نمایش داده شوند
              </p>
            </CardHeader>
            <CardContent>
              <p className="text-center text-gray-500 py-8">
                پیکربندی ویجت به‌زودی...
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bell className="h-5 w-5" />
                تنظیمات اعلان‌ها
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-center text-gray-500 py-8">
                تنظیمات اعلان به‌زودی...
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="display">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Palette className="h-5 w-5" />
                تنظیمات نمایش
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-center text-gray-500 py-8">
                تنظیمات نمایش به‌زودی...
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trading">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                تنظیمات معاملات
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-center text-gray-500 py-8">
                تنظیمات معاملات به‌زودی...
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
