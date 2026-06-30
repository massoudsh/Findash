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
    language: 'en',
    timezone: 'America/New_York',
    dateFormat: 'MM/dd/yyyy',
    numberFormat: 'en-US',
    currency: 'USD',
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
            Settings
          </h1>
          <p className="text-muted-foreground">
            Configure your application preferences and system settings
          </p>
        </div>
        
        <div className="flex gap-3">
          <Button variant="outline" onClick={exportSettings}>
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button variant="outline" onClick={handleResetToDefaults}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Reset
          </Button>
          <Button onClick={handleSaveSettings} disabled={!hasChanges || saving}>
            {saving ? (
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Save className="w-4 h-4 mr-2" />
            )}
            {saving ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>

      {hasChanges && (
        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center gap-2 text-yellow-800">
            <AlertTriangle className="w-5 h-5" />
            <span>You have unsaved changes. Remember to save before leaving this page.</span>
          </div>
        </div>
      )}

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="trading">Trading</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
        </TabsList>

        <TabsContent value="general" className="space-y-4">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Palette className="w-5 h-5" />
                  Appearance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Theme</label>
                  <select 
                    value={appSettings.theme}
                    onChange={(e) => {
                      setAppSettings({...appSettings, theme: e.target.value as any});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="light">Light</option>
                    <option value="dark">Dark</option>
                    <option value="auto">Auto (System)</option>
                  </select>
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">Animations</div>
                    <div className="text-sm text-gray-600">Enable smooth transitions</div>
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
                    <div className="font-medium">Compact Mode</div>
                    <div className="text-sm text-gray-600">Reduce spacing and padding</div>
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
                      Sound Effects
                    </div>
                    <div className="text-sm text-gray-600">Enable audio notifications</div>
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
                  Localization
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Language</label>
                  <select 
                    value={appSettings.language}
                    onChange={(e) => {
                      setAppSettings({...appSettings, language: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="en">English</option>
                    <option value="es">Español</option>
                    <option value="fr">Français</option>
                    <option value="de">Deutsch</option>
                    <option value="ja">日本語</option>
                  </select>
                </div>
                
                <div>
                  <label className="text-sm font-medium">Timezone</label>
                  <select 
                    value={appSettings.timezone}
                    onChange={(e) => {
                      setAppSettings({...appSettings, timezone: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="America/New_York">Eastern Time (ET)</option>
                    <option value="America/Chicago">Central Time (CT)</option>
                    <option value="America/Denver">Mountain Time (MT)</option>
                    <option value="America/Los_Angeles">Pacific Time (PT)</option>
                    <option value="Europe/London">London (GMT)</option>
                    <option value="Europe/Frankfurt">Frankfurt (CET)</option>
                    <option value="Asia/Tokyo">Tokyo (JST)</option>
                  </select>
                </div>
                
                <div>
                  <label className="text-sm font-medium">Date Format</label>
                  <select 
                    value={appSettings.dateFormat}
                    onChange={(e) => {
                      setAppSettings({...appSettings, dateFormat: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="MM/dd/yyyy">MM/dd/yyyy (US)</option>
                    <option value="dd/MM/yyyy">dd/MM/yyyy (EU)</option>
                    <option value="yyyy-MM-dd">yyyy-MM-dd (ISO)</option>
                  </select>
                </div>
                
                <div>
                  <label className="text-sm font-medium">Currency</label>
                  <select 
                    value={appSettings.currency}
                    onChange={(e) => {
                      setAppSettings({...appSettings, currency: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="USD">US Dollar (USD)</option>
                    <option value="EUR">Euro (EUR)</option>
                    <option value="GBP">British Pound (GBP)</option>
                    <option value="JPY">Japanese Yen (JPY)</option>
                    <option value="CAD">Canadian Dollar (CAD)</option>
                  </select>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Performance
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">Auto-save</div>
                    <div className="text-sm text-gray-600">Automatically save changes</div>
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
                    <div className="font-medium">Auto-refresh</div>
                    <div className="text-sm text-gray-600">Automatically refresh data</div>
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
                  <label className="text-sm font-medium">Refresh Interval (seconds)</label>
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
                  Order Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Default Order Type</label>
                  <select 
                    value={tradingSettings.defaultOrderType}
                    onChange={(e) => {
                      setTradingSettings({...tradingSettings, defaultOrderType: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="market">Market</option>
                    <option value="limit">Limit</option>
                    <option value="stop">Stop</option>
                    <option value="stop-limit">Stop Limit</option>
                  </select>
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">تأیید سفارش‌ها</div>
                    <div className="text-sm text-gray-600">Show confirmation dialog</div>
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
                    <div className="font-medium">Risk Warnings</div>
                    <div className="text-sm text-gray-600">Show risk alerts</div>
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
                    <div className="font-medium">Paper Trading Mode</div>
                    <div className="text-sm text-gray-600">Use virtual money</div>
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
                  Risk Management
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Max Positions</label>
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
                  <label className="text-sm font-medium">Default Leverage</label>
                  <select 
                    value={tradingSettings.defaultLeverage}
                    onChange={(e) => {
                      setTradingSettings({...tradingSettings, defaultLeverage: Number(e.target.value)});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="1">1:1 (No Leverage)</option>
                    <option value="2">1:2</option>
                    <option value="5">1:5</option>
                    <option value="10">1:10</option>
                    <option value="20">1:20</option>
                  </select>
                </div>
                
                <div>
                  <label className="text-sm font-medium">Default Stop Loss (%)</label>
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
                  <label className="text-sm font-medium">Default Take Profit (%)</label>
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
                  Notification Channels
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      <Mail className="w-4 h-4" />
                      Email Notifications
                    </div>
                    <div className="text-sm text-gray-600">Receive alerts via email</div>
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
                      Push Notifications
                    </div>
                    <div className="text-sm text-gray-600">Browser push notifications</div>
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
                      SMS Notifications
                    </div>
                    <div className="text-sm text-gray-600">Text message alerts</div>
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
                      Desktop Notifications
                    </div>
                    <div className="text-sm text-gray-600">System desktop alerts</div>
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
                <CardTitle>Alert Types</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">Trade Alerts</div>
                    <div className="text-sm text-gray-600">Order executions and fills</div>
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
                    <div className="font-medium">Price Alerts</div>
                    <div className="text-sm text-gray-600">Price movement notifications</div>
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
                    <div className="font-medium">News Alerts</div>
                    <div className="text-sm text-gray-600">Market news and updates</div>
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
                    <div className="font-medium">System Alerts</div>
                    <div className="text-sm text-gray-600">Platform status updates</div>
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
                Quiet Hours
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="text-sm font-medium">Start Time</label>
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
                  <label className="text-sm font-medium">End Time</label>
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
                Notifications will be silenced during these hours
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
                  Authentication
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">Two-Factor Authentication</div>
                    <div className="text-sm text-gray-600">Enhanced security layer</div>
                  </div>
                  <Badge className={securitySettings.twoFactorAuth ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                    {securitySettings.twoFactorAuth ? 'Enabled' : 'Disabled'}
                  </Badge>
                </div>
                
                <div>
                  <label className="text-sm font-medium">Session Timeout (minutes)</label>
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
                  <label className="text-sm font-medium">Password Expiry (days)</label>
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
                    <div className="font-medium">Login Alerts</div>
                    <div className="text-sm text-gray-600">Notify on new logins</div>
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
                  Access Control
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">IP Restrictions</div>
                    <div className="text-sm text-gray-600">Limit access by IP address</div>
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
                    <div className="font-medium">API Access</div>
                    <div className="text-sm text-gray-600">Allow API key usage</div>
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
                    <div className="font-medium">Audit Logging</div>
                    <div className="text-sm text-gray-600">Log all user actions</div>
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
                    <div className="font-medium">Device Tracking</div>
                    <div className="text-sm text-gray-600">Track login devices</div>
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
                  Data Management
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Data Retention (days)</label>
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
                  <label className="text-sm font-medium">Backup Frequency</label>
                  <select 
                    value={systemSettings.backupFrequency}
                    onChange={(e) => {
                      setSystemSettings({...systemSettings, backupFrequency: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="hourly">Hourly</option>
                    <option value="daily">Daily</option>
                    <option value="weekly">Weekly</option>
                    <option value="monthly">Monthly</option>
                  </select>
                </div>
                
                <div>
                  <label className="text-sm font-medium">Log Level</label>
                  <select 
                    value={systemSettings.logLevel}
                    onChange={(e) => {
                      setSystemSettings({...systemSettings, logLevel: e.target.value});
                      setHasChanges(true);
                    }}
                    className="w-full px-3 py-2 border rounded-md mt-1"
                  >
                    <option value="error">Error</option>
                    <option value="warn">Warning</option>
                    <option value="info">Info</option>
                    <option value="debug">Debug</option>
                  </select>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  Performance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">Performance Mode</div>
                    <div className="text-sm text-gray-600">Optimize for speed</div>
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
                    <div className="font-medium">Cache Enabled</div>
                    <div className="text-sm text-gray-600">Enable data caching</div>
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
                    <div className="font-medium">Compression</div>
                    <div className="text-sm text-gray-600">Compress data transfers</div>
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
                    <div className="font-medium">Analytics</div>
                    <div className="text-sm text-gray-600">Collect usage analytics</div>
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
                Advanced Options
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-orange-600">Maintenance Mode</div>
                    <div className="text-sm text-gray-600">Enable maintenance mode</div>
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
                    <div className="font-medium text-red-600">Debug Mode</div>
                    <div className="text-sm text-gray-600">Enable debug logging</div>
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
                    <div className="font-medium">Warning</div>
                    <div className="text-sm">These advanced options may affect system performance and should only be changed by administrators.</div>
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