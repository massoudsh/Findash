'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  User,
  Settings,
  Shield,
  Key,
  Bell,
  CreditCard,
  Eye,
  EyeOff,
  Edit,
  Save,
  X,
  Check,
  AlertTriangle,
  Download,
  Upload,
  Smartphone,
  Mail,
  Phone,
  MapPin,
  Calendar,
  Clock,
  Globe,
  Lock,
  Unlock,
  Copy,
  RefreshCw,
  Trash2,
  Plus,
  Activity,
  BarChart3,
  DollarSign,
  TrendingUp,
  Zap,
  Target,
  Camera
} from 'lucide-react';

interface UserProfile {
  id: string;
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  avatar: string;
  role: string;
  joinDate: string;
  lastLogin: string;
  location: string;
  timezone: string;
  bio: string;
  verified: boolean;
  twoFactorEnabled: boolean;
  emailNotifications: boolean;
  smsNotifications: boolean;
  pushNotifications: boolean;
}

interface TradingPreferences {
  defaultOrderType: string;
  defaultTimeInForce: string;
  riskTolerance: string;
  maxPositionSize: number;
  stopLossDefault: number;
  takeProfitDefault: number;
  autoConfirmOrders: boolean;
  showAdvancedFeatures: boolean;
  preferredMarkets: string[];
  tradingHours: { start: string; end: string };
}

interface APIKey {
  id: string;
  name: string;
  key: string;
  permissions: string[];
  created: string;
  lastUsed: string;
  status: 'active' | 'inactive';
}

interface SecurityLog {
  id: string;
  action: string;
  timestamp: string;
  ipAddress: string;
  location: string;
  device: string;
  status: 'success' | 'failed';
}

export default function ProfilePage() {
  const [selectedTab, setSelectedTab] = useState('personal');
  const [isEditing, setIsEditing] = useState(false);
  const [showApiKey, setShowApiKey] = useState<string | null>(null);

  const [profile, setProfile] = useState<UserProfile>({
    id: 'user-123',
    firstName: 'مسعود',
    lastName: 'شمیرانی',
    email: 'massoud.shemirani@company.com',
    phone: '+98 912 123 4567',
    avatar: '',
    role: 'معامله‌گر ارشد',
    joinDate: '2022-03-15',
    lastLogin: '2024-01-20T15:45:23Z',
    location: 'تهران، ایران',
    timezone: 'تهران (UTC+3:30)',
    bio: 'معامله‌گر کمی با تخصص در استراتژی‌های الگوریتمی و مدیریت ریسک.',
    verified: true,
    twoFactorEnabled: true,
    emailNotifications: true,
    smsNotifications: false,
    pushNotifications: true
  });

  const [tradingPrefs, setTradingPrefs] = useState<TradingPreferences>({
    defaultOrderType: 'limit',
    defaultTimeInForce: 'day',
    riskTolerance: 'moderate',
    maxPositionSize: 100000,
    stopLossDefault: 2.0,
    takeProfitDefault: 4.0,
    autoConfirmOrders: false,
    showAdvancedFeatures: true,
    preferredMarkets: ['stocks', 'crypto', 'forex'],
    tradingHours: { start: '09:30', end: '16:00' }
  });

  const [apiKeys, setApiKeys] = useState<APIKey[]>([
    {
      id: 'api-001',
      name: 'ربات معاملاتی نسخه ۱',
      key: 'ak_live_123abc...def789',
      permissions: ['read', 'trade'],
      created: '2024-01-15',
      lastUsed: '2024-01-20T10:30:00Z',
      status: 'active'
    },
    {
      id: 'api-002',
      name: 'تحلیل داده',
      key: 'ak_live_456ghi...jkl012',
      permissions: ['read'],
      created: '2024-01-10',
      lastUsed: '2024-01-19T14:15:00Z',
      status: 'active'
    }
  ]);

  const [securityLogs, setSecurityLogs] = useState<SecurityLog[]>([
    {
      id: 'sec-001',
      action: 'ورود',
      timestamp: '2024-01-20T15:45:23Z',
      ipAddress: '192.168.1.145',
      location: 'تهران',
      device: 'Chrome روی Windows',
      status: 'success'
    },
    {
      id: 'sec-002',
      action: 'تغییر رمز عبور',
      timestamp: '2024-01-18T09:30:15Z',
      ipAddress: '192.168.1.145',
      location: 'تهران',
      device: 'Chrome روی Windows',
      status: 'success'
    },
    {
      id: 'sec-003',
      action: 'ورود ناموفق',
      timestamp: '2024-01-17T23:15:42Z',
      ipAddress: '203.45.67.89',
      location: 'نامشخص',
      device: 'نامشخص',
      status: 'failed'
    }
  ]);

  const handleSaveProfile = () => {
    // Save profile logic
    setIsEditing(false);
  };

  const handleGenerateApiKey = () => {
    const newKey: APIKey = {
      id: `api-${Date.now()}`,
      name: `کلید API جدید ${apiKeys.length + 1}`,
      key: `ak_live_${Math.random().toString(36).substring(2, 15)}...`,
      permissions: ['read'],
      created: new Date().toISOString().split('T')[0],
      lastUsed: 'هرگز',
      status: 'active'
    };
    setApiKeys([...apiKeys, newKey]);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('fa-IR');
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <User className="w-8 h-8 text-blue-600" />
            پروفایل کاربری
          </h1>
          <p className="text-muted-foreground">
            مدیریت تنظیمات حساب، امنیت و ترجیحات معاملاتی
          </p>
        </div>

        <div className="flex gap-3">
          {isEditing ? (
            <>
              <Button variant="outline" onClick={() => setIsEditing(false)}>
                <X className="w-4 h-4 mr-2" />
                انصراف
              </Button>
              <Button onClick={handleSaveProfile}>
                <Save className="w-4 h-4 mr-2" />
                ذخیره تغییرات
              </Button>
            </>
          ) : (
            <Button onClick={() => setIsEditing(true)}>
              <Edit className="w-4 h-4 mr-2" />
              ویرایش پروفایل
            </Button>
          )}
        </div>
      </div>

      {/* Profile Header */}
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center gap-6">
            <div className="relative">
              <div className="w-24 h-24 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white text-2xl font-bold">
                {profile.firstName[0]}{profile.lastName[0]}
              </div>
              {isEditing && (
                <Button size="sm" className="absolute -bottom-2 -right-2 rounded-full w-8 h-8 p-0">
                  <Camera className="w-4 h-4" />
                </Button>
              )}
            </div>

            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <h2 className="text-2xl font-bold">{profile.firstName} {profile.lastName}</h2>
                {profile.verified && (
                  <Badge className="bg-green-100 text-green-800">
                    <Check className="w-3 h-3 mr-1" />
                    تأییدشده
                  </Badge>
                )}
                <Badge variant="outline">{profile.role}</Badge>
              </div>
              <p className="text-gray-600 mb-2">{profile.email}</p>
              <div className="flex items-center gap-4 text-sm text-gray-500">
                <span className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  عضویت از {new Date(profile.joinDate).toLocaleDateString('fa-IR')}
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  آخرین ورود {formatTimestamp(profile.lastLogin)}
                </span>
                <span className="flex items-center gap-1">
                  <MapPin className="w-4 h-4" />
                  {profile.location}
                </span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="personal">اطلاعات شخصی</TabsTrigger>
          <TabsTrigger value="security">امنیت</TabsTrigger>
          <TabsTrigger value="trading">معاملات</TabsTrigger>
          <TabsTrigger value="api">کلیدهای API</TabsTrigger>
          <TabsTrigger value="notifications">اعلان‌ها</TabsTrigger>
        </TabsList>

        <TabsContent value="personal" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>اطلاعات شخصی</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="text-sm font-medium">نام</label>
                  <Input
                    value={profile.firstName}
                    onChange={(e) => setProfile({...profile, firstName: e.target.value})}
                    disabled={!isEditing}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">نام خانوادگی</label>
                  <Input
                    value={profile.lastName}
                    onChange={(e) => setProfile({...profile, lastName: e.target.value})}
                    disabled={!isEditing}
                  />
                </div>
              </div>

              <div>
                <label className="text-sm font-medium">آدرس ایمیل</label>
                <Input
                  value={profile.email}
                  onChange={(e) => setProfile({...profile, email: e.target.value})}
                  disabled={!isEditing}
                />
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="text-sm font-medium">شماره تلفن</label>
                  <Input
                    value={profile.phone}
                    onChange={(e) => setProfile({...profile, phone: e.target.value})}
                    disabled={!isEditing}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">منطقه زمانی</label>
                  <Input
                    value={profile.timezone}
                    onChange={(e) => setProfile({...profile, timezone: e.target.value})}
                    disabled={!isEditing}
                  />
                </div>
              </div>

              <div>
                <label className="text-sm font-medium">درباره من</label>
                <textarea
                  className="w-full p-3 border rounded-md resize-none"
                  rows={3}
                  value={profile.bio}
                  onChange={(e) => setProfile({...profile, bio: e.target.value})}
                  disabled={!isEditing}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  تنظیمات امنیتی
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">تأیید دومرحله‌ای</div>
                    <div className="text-sm text-gray-600">افزودن یک لایه امنیتی اضافه</div>
                  </div>
                  <div className="flex items-center gap-2">
                    {profile.twoFactorEnabled ? (
                      <Badge className="bg-green-100 text-green-800">فعال</Badge>
                    ) : (
                      <Badge variant="outline">غیرفعال</Badge>
                    )}
                    <Button size="sm" variant="outline">
                      {profile.twoFactorEnabled ? 'غیرفعال‌سازی' : 'فعال‌سازی'}
                    </Button>
                  </div>
                </div>

                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">رمز عبور</div>
                    <div className="text-sm text-gray-600">آخرین تغییر ۳ روز پیش</div>
                  </div>
                  <Button size="sm" variant="outline">
                    تغییر رمز عبور
                  </Button>
                </div>

                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">جلسات ورود</div>
                    <div className="text-sm text-gray-600">مدیریت جلسات فعال</div>
                  </div>
                  <Button size="sm" variant="outline">
                    مشاهده جلسات
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="w-5 h-5" />
                  فعالیت‌های اخیر
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {securityLogs.slice(0, 4).map((log) => (
                    <div key={log.id} className="flex items-center justify-between p-2 border rounded">
                      <div className="flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full ${
                          log.status === 'success' ? 'bg-green-500' : 'bg-red-500'
                        }`} />
                        <div>
                          <div className="text-sm font-medium">{log.action}</div>
                          <div className="text-xs text-gray-600">
                            {formatTimestamp(log.timestamp)} • {log.location}
                          </div>
                        </div>
                      </div>
                      <Badge className={log.status === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                        {log.status === 'success' ? 'موفق' : 'ناموفق'}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="trading" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                ترجیحات معاملاتی
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="text-sm font-medium">نوع سفارش پیش‌فرض</label>
                  <select
                    value={tradingPrefs.defaultOrderType}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, defaultOrderType: e.target.value})}
                    className="w-full px-3 py-2 border rounded-md"
                  >
                    <option value="market">بازار</option>
                    <option value="limit">محدود (Limit)</option>
                    <option value="stop">استاپ</option>
                    <option value="stop-limit">استاپ محدود</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium">میزان ریسک‌پذیری</label>
                  <select
                    value={tradingPrefs.riskTolerance}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, riskTolerance: e.target.value})}
                    className="w-full px-3 py-2 border rounded-md"
                  >
                    <option value="conservative">محافظه‌کار</option>
                    <option value="moderate">متعادل</option>
                    <option value="aggressive">تهاجمی</option>
                  </select>
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                <div>
                  <label className="text-sm font-medium">حداکثر حجم پوزیشن (تومان)</label>
                  <Input
                    type="number"
                    value={tradingPrefs.maxPositionSize}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, maxPositionSize: Number(e.target.value)})}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">حد ضرر پیش‌فرض (%)</label>
                  <Input
                    type="number"
                    step="0.1"
                    value={tradingPrefs.stopLossDefault}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, stopLossDefault: Number(e.target.value)})}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">حد سود پیش‌فرض (%)</label>
                  <Input
                    type="number"
                    step="0.1"
                    value={tradingPrefs.takeProfitDefault}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, takeProfitDefault: Number(e.target.value)})}
                  />
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">تأیید خودکار سفارش‌ها</div>
                    <div className="text-sm text-gray-600">ثبت خودکار سفارش‌ها بدون تأیید دستی</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={tradingPrefs.autoConfirmOrders}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, autoConfirmOrders: e.target.checked})}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">نمایش امکانات پیشرفته</div>
                    <div className="text-sm text-gray-600">نمایش ابزارهای معاملاتی پیشرفته</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={tradingPrefs.showAdvancedFeatures}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, showAdvancedFeatures: e.target.checked})}
                    className="w-4 h-4"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="api" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle className="flex items-center gap-2">
                  <Key className="w-5 h-5" />
                  کلیدهای API
                </CardTitle>
                <Button onClick={handleGenerateApiKey}>
                  <Plus className="w-4 h-4 mr-2" />
                  ایجاد کلید جدید
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {apiKeys.map((apiKey) => (
                  <div key={apiKey.id} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <div className="font-medium">{apiKey.name}</div>
                        <div className="text-sm text-gray-600">
                          ایجاد {apiKey.created} • آخرین استفاده {apiKey.lastUsed}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge className={apiKey.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}>
                          {apiKey.status === 'active' ? 'فعال' : 'غیرفعال'}
                        </Badge>
                        <Button size="sm" variant="outline" onClick={() => setShowApiKey(showApiKey === apiKey.id ? null : apiKey.id)}>
                          {showApiKey === apiKey.id ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        </Button>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 mb-3">
                      <Input
                        value={showApiKey === apiKey.id ? apiKey.key : '••••••••••••••••••••'}
                        readOnly
                        className="font-mono text-sm"
                      />
                      <Button size="sm" variant="outline" onClick={() => copyToClipboard(apiKey.key)}>
                        <Copy className="w-4 h-4" />
                      </Button>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex gap-2">
                        {apiKey.permissions.map(permission => (
                          <Badge key={permission} variant="outline">
                            {permission}
                          </Badge>
                        ))}
                      </div>
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline">
                          <RefreshCw className="w-4 h-4" />
                        </Button>
                        <Button size="sm" variant="outline">
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bell className="w-5 h-5" />
                ترجیحات اعلان
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      <Mail className="w-4 h-4" />
                      اعلان‌های ایمیلی
                    </div>
                    <div className="text-sm text-gray-600">دریافت اعلان از طریق ایمیل</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={profile.emailNotifications}
                    onChange={(e) => setProfile({...profile, emailNotifications: e.target.checked})}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      <Smartphone className="w-4 h-4" />
                      اعلان‌های پیامکی
                    </div>
                    <div className="text-sm text-gray-600">دریافت اعلان از طریق پیامک</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={profile.smsNotifications}
                    onChange={(e) => setProfile({...profile, smsNotifications: e.target.checked})}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      <Bell className="w-4 h-4" />
                      اعلان‌های پوش
                    </div>
                    <div className="text-sm text-gray-600">دریافت اعلان‌های پوش مرورگر</div>
                  </div>
                  <input
                    type="checkbox"
                    checked={profile.pushNotifications}
                    onChange={(e) => setProfile({...profile, pushNotifications: e.target.checked})}
                    className="w-4 h-4"
                  />
                </div>
              </div>

              <div className="pt-4 border-t">
                <h3 className="font-medium mb-3">انواع اعلان</h3>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked className="w-4 h-4" />
                      <span className="text-sm">اجرای معاملات</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked className="w-4 h-4" />
                      <span className="text-sm">هشدار قیمت</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked className="w-4 h-4" />
                      <span className="text-sm">هشدار ریسک</span>
                    </label>
                  </div>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked className="w-4 h-4" />
                      <span className="text-sm">به‌روزرسانی بازار</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="w-4 h-4" />
                      <span className="text-sm">تعمیر و نگهداری سیستم</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="w-4 h-4" />
                      <span className="text-sm">امنیت حساب</span>
                    </label>
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