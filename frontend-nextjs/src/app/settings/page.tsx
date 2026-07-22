'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Settings,
  Monitor,
  Globe,
  Shield,
  Bell,
  Palette,
  Database,
  Cloud,
  Download,
  Upload,
  Save,
  RefreshCw,
  AlertTriangle,
  Check,
  X,
  Moon,
  Sun,
  Volume2,
  VolumeX,
  Zap,
  Timer,
  Lock,
  Unlock,
  Eye,
  EyeOff,
  Key,
  FileText,
  BarChart3,
  TrendingUp,
  Activity,
  Target,
  Users,
  Building,
  CreditCard,
  Smartphone,
  Mail,
  Phone,
  MapPin,
  Calendar,
  Clock,
  Edit,
  Trash2,
  Plus,
  Minus,
  Copy,
  ExternalLink,
  HelpCircle,
  Info,
  Search,
  Filter,
  SortAsc,
  Archive,
  Star
} from 'lucide-react';
import { UserPreferences } from '@/components/ui/user-preferences';

interface AppSettings {
  theme: 'light' | 'dark' | 'auto';
  language: string;
  timezone: string;
  dateFormat: string;
  numberFormat: string;
  currency: string;
  soundEnabled: boolean;
  animationsEnabled: boolean;
  autoSave: boolean;
  autoRefresh: boolean;
  refreshInterval: number;
  compactMode: boolean;
}

interface TradingSettings {
  defaultOrderType: string;
  confirmOrders: boolean;
  showAdvanced: boolean;
  riskWarnings: boolean;
  positionSizing: boolean;
  paperTradingMode: boolean;
  maxPositions: number;
  defaultLeverage: number;
  stopLossDefault: number;
  takeProfitDefault: number;
}

interface NotificationSettings {
  emailNotifications: boolean;
  pushNotifications: boolean;
  smsNotifications: boolean;
  desktopNotifications: boolean;
  tradeAlerts: boolean;
  priceAlerts: boolean;
  newsAlerts: boolean;
  systemAlerts: boolean;
  maintenanceAlerts: boolean;
  quietHours: { start: string; end: string };
}

interface SecuritySettings {
  twoFactorAuth: boolean;
  sessionTimeout: number;
  loginAlerts: boolean;
  ipRestrictions: boolean;
  allowedIPs: string[];
  apiAccess: boolean;
  auditLogging: boolean;
  passwordExpiry: number;
  deviceTracking: boolean;
}

interface SystemSettings {
  dataRetention: number;
  backupFrequency: string;
  logLevel: string;
  performanceMode: boolean;
  cacheEnabled: boolean;
  compressionEnabled: boolean;
  maintenanceMode: boolean;
  debugMode: boolean;
  analyticsEnabled: boolean;
}

export default function SettingsPage() {
  const [selectedTab, setSelectedTab] = useState('general');
  const [hasChanges, setHasChanges] = useState(false);
  const [saving, setSaving] = useState(false);

  const [appSettings, setAppSettings] = useState<AppSettings>({
    theme: 'dark',
    language: 'fa',
    timezone: 'Asia/Tehran',
    dateFormat: 'yyyy-MM-dd',
    numberFormat: 'fa-IR',
    currency: 'IRT',
    soundEnabled: true,
    animationsEnabled: true,
    autoSave: true,
    autoRefresh: true,
    refreshInterval: 5,
    compactMode: false
  });

  const [tradingSettings, setTradingSettings] = useState<TradingSettings>({
    defaultOrderType: 'limit',
    confirmOrders: true,
    showAdvanced: true,
    riskWarnings: true,
    positionSizing: true,
    paperTradingMode: false,
    maxPositions: 10,
    defaultLeverage: 1,
    stopLossDefault: 2.0,
    takeProfitDefault: 4.0
  });

  const [notificationSettings, setNotificationSettings] = useState<NotificationSettings>({
    emailNotifications: true,
    pushNotifications: true,
    smsNotifications: false,
    desktopNotifications: true,
    tradeAlerts: true,
    priceAlerts: true,
    newsAlerts: false,
    systemAlerts: true,
    maintenanceAlerts: true,
    quietHours: { start: '22:00', end: '08:00' }
  });

  const [securitySettings, setSecuritySettings] = useState<SecuritySettings>({
    twoFactorAuth: true,
    sessionTimeout: 30,
    loginAlerts: true,
    ipRestrictions: false,
    allowedIPs: ['192.168.1.0/24'],
    apiAccess: true,
    auditLogging: true,
    passwordExpiry: 90,
    deviceTracking: true
  });

  const [systemSettings, setSystemSettings] = useState<SystemSettings>({
    dataRetention: 365,
    backupFrequency: 'daily',
    logLevel: 'info',
    performanceMode: true,
    cacheEnabled: true,
    compressionEnabled: true,
    maintenanceMode: false,
    debugMode: false,
    analyticsEnabled: true
  });

  const handleSaveSettings = async () => {
    setSaving(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    setHasChanges(false);
    setSaving(false);
  };

  const handleResetToDefaults = () => {
    // Reset all settings to defaults
    setHasChanges(true);
  };

  const exportSettings = () => {
    const allSettings = {
      app: appSettings,
      trading: tradingSettings,
      notifications: notificationSettings,
      security: securitySettings,
      system: systemSettings
    };

    const blob = new Blob([JSON.stringify(allSettings, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'trading-platform-settings.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <Settings className="w-8 h-8 text-blue-600" />
            تنظیمات
          </h1>
          <p className="text-muted-foreground">
            تنظیمات برنامه و پیکربندی سیستم را مدیریت کنید
          </p>
        </div>

        <div className="flex gap-3">
          <Button variant="outline" onClick={exportSettings}>
            <Download className="w-4 h-4 mr-2" />
            خروجی
          </Button>
          <Button variant="outline" onClick={handleResetToDefaults}>
            <RefreshCw className="w-4 h-4 mr-2" />
            بازنشانی
          </Button>
          <Button onClick={handleSaveSettings} disabled={!hasChanges || saving}>
            {saving ? (
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Save className="w-4 h-4 mr-2" />
            )}
            {saving ? 'در حال ذخیره...' : 'ذخیره تغییرات'}
          </Button>
        </div>
      </div>

      {hasChanges && (
        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center gap-2 text-yellow-800">
            <AlertTriangle className="w-5 h-5" />
            <span>تغییرات ذخیره‌نشده دارید. قبل از خروج از این صفحه، آن‌ها را ذخیره کنید.</span>
          </div>
        </div>
      )}

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="general">عمومی</TabsTrigger>
          <TabsTrigger value="trading">معاملات</TabsTrigger>
          <TabsTrigger value="notifications">اعلان‌ها</TabsTrigger>
          <TabsTrigger value="security">امنیت</TabsTrigger>
          <TabsTrigger value="system">سیستم</TabsTrigger>
        </TabsList>

        <TabsContent value="general" className="space-y-4">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Palette className="w-5 h-5" />
                  ظاهر
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">پوسته</label>
                  <select
                    value={appSettings.theme}
                    onChange={(e) => {
                      setAppSettings({...appSettings, theme: e.target.value as any});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="light">روشن</option>
                    <option value="dark">تیره</option>
                    <option value="auto">خودکار (سیستم)</option>
                  </select>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">انیمیشن‌ها</div>
                    <div className="text-sm text-gray-600">فعال‌سازی انتقال‌های نرم</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={appSettings.animationsEnabled}
                    onChange={(e) => {
                      setAppSettings({...appSettings, animationsEnabled: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">حالت فشرده</div>
                    <div className="text-sm text-gray-600">کاهش فاصله‌گذاری و پدینگ</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={appSettings.compactMode}
                    onChange={(e) => {
                      setAppSettings({...appSettings, compactMode: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      {appSettings.soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
                      جلوه‌های صوتی
                    </div>
                    <div className="text-sm text-gray-600">فعال‌سازی اعلان‌های صوتی</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={appSettings.soundEnabled}
                    onChange={(e) => {
                      setAppSettings({...appSettings, soundEnabled: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Globe className="w-5 h-5" />
                  محلی‌سازی
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">زبان</label>
                  <select
                    value={appSettings.language}
                    onChange={(e) => {
                      setAppSettings({...appSettings, language: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="fa">فارسی</option>
                    <option value="en">English</option>
                    <option value="es">Español</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium">منطقه زمانی</label>
                  <select
                    value={appSettings.timezone}
                    onChange={(e) => {
                      setAppSettings({...appSettings, timezone: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="Asia/Tehran">تهران (UTC+3:30)</option>
                    <option value="America/New_York">وقت شرقی آمریکا (ET)</option>
                    <option value="America/Chicago">وقت مرکزی آمریکا (CT)</option>
                    <option value="Europe/London">لندن (GMT)</option>
                    <option value="Europe/Frankfurt">فرانکفورت (CET)</option>
                    <option value="Asia/Tokyo">توکیو (JST)</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium">قالب تاریخ</label>
                  <select
                    value={appSettings.dateFormat}
                    onChange={(e) => {
                      setAppSettings({...appSettings, dateFormat: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="yyyy-MM-dd">yyyy-MM-dd (شمسی/میلادی ISO)</option>
                    <option value="MM/dd/yyyy">MM/dd/yyyy (آمریکایی)</option>
                    <option value="dd/MM/yyyy">dd/MM/yyyy (اروپایی)</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium">واحد پول</label>
                  <select
                    value={appSettings.currency}
                    onChange={(e) => {
                      setAppSettings({...appSettings, currency: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="IRT">تومان (IRT)</option>
                    <option value="USD">دلار آمریکا (USD)</option>
                    <option value="EUR">یورو (EUR)</option>
                    <option value="GBP">پوند انگلیس (GBP)</option>
                  </select>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                عملکرد
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">ذخیره خودکار</div>
                    <div className="text-sm text-gray-600">ذخیره خودکار تغییرات</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={appSettings.autoSave}
                    onChange={(e) => {
                      setAppSettings({...appSettings, autoSave: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">به‌روزرسانی خودکار</div>
                    <div className="text-sm text-gray-600">به‌روزرسانی خودکار داده‌ها</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={appSettings.autoRefresh}
                    onChange={(e) => {
                      setAppSettings({...appSettings, autoRefresh: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium">بازه به‌روزرسانی (ثانیه)</label>
                  <Input
                    type="number"
                    min="1"
                    max="60"
                    value={appSettings.refreshInterval}
                    onChange={(e) => {
                      setAppSettings({...appSettings, refreshInterval: Number(e.target.value)});
                      setHasChanges(true);
                    }}
                    disabled={!appSettings.autoRefresh}
                    className="mt-1"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trading" className="space-y-4">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5" />
                  تنظیمات سفارش
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">نوع سفارش پیش‌فرض</label>
                  <select
                    value={tradingSettings.defaultOrderType}
                    onChange={(e) => {
                      setTradingSettings({...tradingSettings, defaultOrderType: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="market">بازار</option>
                    <option value="limit">محدود</option>
                    <option value="stop">استاپ</option>
                    <option value="stop-limit">استاپ محدود</option>
                  </select>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">تأیید سفارش‌ها</div>
                    <div className="text-sm text-gray-600">نمایش دیالوگ تأیید</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={tradingSettings.confirmOrders}
                    onChange={(e) => {
                      setTradingSettings({...tradingSettings, confirmOrders: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">هشدارهای ریسک</div>
                    <div className="text-sm text-gray-600">نمایش هشدارهای ریسک</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={tradingSettings.riskWarnings}
                    onChange={(e) => {
                      setTradingSettings({...tradingSettings, riskWarnings: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">حالت معاملات آزمایشی</div>
                    <div className="text-sm text-gray-600">استفاده از پول مجازی</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={tradingSettings.paperTradingMode}
                    onChange={(e) => {
                      setTradingSettings({...tradingSettings, paperTradingMode: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5" />
                  مدیریت ریسک
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">حداکثر پوزیشن‌ها</label>
                  <Input
                    type="number"
                    min="1"
                    max="50"
                    value={tradingSettings.maxPositions}
                    onChange={(e) => {
                      setTradingSettings({...tradingSettings, maxPositions: Number(e.target.value)});
                      setHasChanges(true);
                    }}
                    className="mt-1"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium">اهرم پیش‌فرض</label>
                  <select
                    value={tradingSettings.defaultLeverage}
                    onChange={(e) => {
                      setTradingSettings({...tradingSettings, defaultLeverage: Number(e.target.value)});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="1">۱:۱ (بدون اهرم)</option>
                    <option value="2">۱:۲</option>
                    <option value="5">۱:۵</option>
                    <option value="10">۱:۱۰</option>
                    <option value="20">۱:۲۰</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium">حد ضرر پیش‌فرض (٪)</label>
                  <Input
                    type="number"
                    min="0.1"
                    max="10"
                    step="0.1"
                    value={tradingSettings.stopLossDefault}
                    onChange={(e) => {
                      setTradingSettings({...tradingSettings, stopLossDefault: Number(e.target.value)});
                      setHasChanges(true);
                    }}
                    className="mt-1"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium">حد سود پیش‌فرض (٪)</label>
                  <Input
                    type="number"
                    min="0.1"
                    max="20"
                    step="0.1"
                    value={tradingSettings.takeProfitDefault}
                    onChange={(e) => {
                      setTradingSettings({...tradingSettings, takeProfitDefault: Number(e.target.value)});
                      setHasChanges(true);
                    }}
                    className="mt-1"
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="notifications" className="space-y-4">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bell className="w-5 h-5" />
                  کانال‌های اعلان
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      <Mail className="w-4 h-4" />
                      اعلان‌های ایمیلی
                    </div>
                    <div className="text-sm text-gray-600">دریافت هشدارها از طریق ایمیل</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={notificationSettings.emailNotifications}
                    onChange={(e) => {
                      setNotificationSettings({...notificationSettings, emailNotifications: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      <Smartphone className="w-4 h-4" />
                      اعلان‌های پوش
                    </div>
                    <div className="text-sm text-gray-600">اعلان‌های پوش مرورگر</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={notificationSettings.pushNotifications}
                    onChange={(e) => {
                      setNotificationSettings({...notificationSettings, pushNotifications: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      <Phone className="w-4 h-4" />
                      پیامک
                    </div>
                    <div className="text-sm text-gray-600">هشدار از طریق پیامک</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={notificationSettings.smsNotifications}
                    onChange={(e) => {
                      setNotificationSettings({...notificationSettings, smsNotifications: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      <Monitor className="w-4 h-4" />
                      اعلان‌های دسکتاپ
                    </div>
                    <div className="text-sm text-gray-600">هشدارهای دسکتاپ سیستم</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={notificationSettings.desktopNotifications}
                    onChange={(e) => {
                      setNotificationSettings({...notificationSettings, desktopNotifications: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>انواع هشدار</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">هشدار معاملات</div>
                    <div className="text-sm text-gray-600">اجرا و تکمیل سفارش‌ها</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={notificationSettings.tradeAlerts}
                    onChange={(e) => {
                      setNotificationSettings({...notificationSettings, tradeAlerts: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">هشدار قیمت</div>
                    <div className="text-sm text-gray-600">اعلان تغییرات قیمت</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={notificationSettings.priceAlerts}
                    onChange={(e) => {
                      setNotificationSettings({...notificationSettings, priceAlerts: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">هشدار اخبار</div>
                    <div className="text-sm text-gray-600">اخبار و به‌روزرسانی‌های بازار</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={notificationSettings.newsAlerts}
                    onChange={(e) => {
                      setNotificationSettings({...notificationSettings, newsAlerts: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">هشدار سیستم</div>
                    <div className="text-sm text-gray-600">به‌روزرسانی وضعیت پلتفرم</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={notificationSettings.systemAlerts}
                    onChange={(e) => {
                      setNotificationSettings({...notificationSettings, systemAlerts: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="w-5 h-5" />
                ساعات سکوت
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="text-sm font-medium">زمان شروع</label>
                  <Input
                    type="time"
                    value={notificationSettings.quietHours.start}
                    onChange={(e) => {
                      setNotificationSettings({
                        ...notificationSettings,
                        quietHours: {...notificationSettings.quietHours, start: e.target.value}
                      });
                      setHasChanges(true);
                    }}
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">زمان پایان</label>
                  <Input
                    type="time"
                    value={notificationSettings.quietHours.end}
                    onChange={(e) => {
                      setNotificationSettings({
                        ...notificationSettings,
                        quietHours: {...notificationSettings.quietHours, end: e.target.value}
                      });
                      setHasChanges(true);
                    }}
                    className="mt-1"
                  />
                </div>
              </div>
              <p className="text-sm text-gray-600 mt-2">
                در این بازه زمانی اعلان‌ها بی‌صدا خواهند شد
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="space-y-4">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  احراز هویت
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">تأیید دومرحله‌ای</div>
                    <div className="text-sm text-gray-600">لایه امنیتی افزوده</div>
                  </div>
                  <Badge className={securitySettings.twoFactorAuth ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                    {securitySettings.twoFactorAuth ? 'فعال' : 'غیرفعال'}
                  </Badge>
                </div>

                <div>
                  <label className="text-sm font-medium">مهلت زمانی نشست (دقیقه)</label>
                  <Input
                    type="number"
                    min="5"
                    max="480"
                    value={securitySettings.sessionTimeout}
                    onChange={(e) => {
                      setSecuritySettings({...securitySettings, sessionTimeout: Number(e.target.value)});
                      setHasChanges(true);
                    }}
                    className="mt-1"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium">انقضای رمز عبور (روز)</label>
                  <Input
                    type="number"
                    min="30"
                    max="365"
                    value={securitySettings.passwordExpiry}
                    onChange={(e) => {
                      setSecuritySettings({...securitySettings, passwordExpiry: Number(e.target.value)});
                      setHasChanges(true);
                    }}
                    className="mt-1"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">هشدار ورود</div>
                    <div className="text-sm text-gray-600">اطلاع‌رسانی ورودهای جدید</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={securitySettings.loginAlerts}
                    onChange={(e) => {
                      setSecuritySettings({...securitySettings, loginAlerts: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Lock className="w-5 h-5" />
                  کنترل دسترسی
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">محدودیت IP</div>
                    <div className="text-sm text-gray-600">محدود کردن دسترسی بر اساس آدرس IP</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={securitySettings.ipRestrictions}
                    onChange={(e) => {
                      setSecuritySettings({...securitySettings, ipRestrictions: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">دسترسی API</div>
                    <div className="text-sm text-gray-600">مجاز بودن استفاده از کلید API</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={securitySettings.apiAccess}
                    onChange={(e) => {
                      setSecuritySettings({...securitySettings, apiAccess: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">ثبت لاگ ممیزی</div>
                    <div className="text-sm text-gray-600">ثبت همه فعالیت‌های کاربر</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={securitySettings.auditLogging}
                    onChange={(e) => {
                      setSecuritySettings({...securitySettings, auditLogging: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">ردیابی دستگاه</div>
                    <div className="text-sm text-gray-600">ردیابی دستگاه‌های ورود</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={securitySettings.deviceTracking}
                    onChange={(e) => {
                      setSecuritySettings({...securitySettings, deviceTracking: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="system" className="space-y-4">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="w-5 h-5" />
                  مدیریت داده
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">نگه‌داری داده (روز)</label>
                  <Input
                    type="number"
                    min="30"
                    max="2555"
                    value={systemSettings.dataRetention}
                    onChange={(e) => {
                      setSystemSettings({...systemSettings, dataRetention: Number(e.target.value)});
                      setHasChanges(true);
                    }}
                    className="mt-1"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium">تناوب پشتیبان‌گیری</label>
                  <select
                    value={systemSettings.backupFrequency}
                    onChange={(e) => {
                      setSystemSettings({...systemSettings, backupFrequency: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="hourly">ساعتی</option>
                    <option value="daily">روزانه</option>
                    <option value="weekly">هفتگی</option>
                    <option value="monthly">ماهانه</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium">سطح لاگ</label>
                  <select
                    value={systemSettings.logLevel}
                    onChange={(e) => {
                      setSystemSettings({...systemSettings, logLevel: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="error">خطا</option>
                    <option value="warn">هشدار</option>
                    <option value="info">اطلاعات</option>
                    <option value="debug">دیباگ</option>
                  </select>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  عملکرد
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">حالت عملکرد بالا</div>
                    <div className="text-sm text-gray-600">بهینه‌سازی برای سرعت</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={systemSettings.performanceMode}
                    onChange={(e) => {
                      setSystemSettings({...systemSettings, performanceMode: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">کش فعال</div>
                    <div className="text-sm text-gray-600">فعال‌سازی کش داده</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={systemSettings.cacheEnabled}
                    onChange={(e) => {
                      setSystemSettings({...systemSettings, cacheEnabled: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">فشرده‌سازی</div>
                    <div className="text-sm text-gray-600">فشرده‌سازی انتقال داده‌ها</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={systemSettings.compressionEnabled}
                    onChange={(e) => {
                      setSystemSettings({...systemSettings, compressionEnabled: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">تحلیل مصرف</div>
                    <div className="text-sm text-gray-600">جمع‌آوری آمار استفاده</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={systemSettings.analyticsEnabled}
                    onChange={(e) => {
                      setSystemSettings({...systemSettings, analyticsEnabled: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                گزینه‌های پیشرفته
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-orange-600">حالت تعمیر و نگهداری</div>
                    <div className="text-sm text-gray-600">فعال‌سازی حالت تعمیر و نگهداری</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={systemSettings.maintenanceMode}
                    onChange={(e) => {
                      setSystemSettings({...systemSettings, maintenanceMode: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-red-600">حالت دیباگ</div>
                    <div className="text-sm text-gray-600">فعال‌سازی لاگ‌های دیباگ</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={systemSettings.debugMode}
                    onChange={(e) => {
                      setSystemSettings({...systemSettings, debugMode: e.target.checked});
                      setHasChanges(true);
                    }}
                    className="w-4 h-4"
                  />
                </div>
              </div>

              <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                <div className="flex items-start gap-2 text-yellow-800">
                  <AlertTriangle className="w-5 h-5 mt-0.5" />
                  <div>
                    <div className="font-medium">هشدار</div>
                    <div className="text-sm">این گزینه‌های پیشرفته می‌توانند بر عملکرد سیستم اثر بگذارند و فقط باید توسط مدیران تغییر داده شوند.</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
