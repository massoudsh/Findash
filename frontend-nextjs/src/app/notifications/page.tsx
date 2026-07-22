'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Bell,
  AlertTriangle,
  CheckCircle,
  XCircle,
  AlertCircle,
  Settings,
  Filter,
  Search,
  Mail,
  MessageSquare,
  Smartphone,
  Volume2,
  VolumeX,
  Eye,
  EyeOff,
  Trash2,
  Archive,
  Clock,
  TrendingUp,
  TrendingDown,
  Shield,
  DollarSign,
  Activity,
  Server,
  Database,
  Wifi,
  WifiOff,
  Zap,
  Target,
  BarChart3,
  User,
  Calendar,
  Download,
  Upload,
  RefreshCw,
  Play,
  Pause,
  StopCircle,
  Cpu,
  HardDrive,
  Network,
  AlertOctagon,
  Info,
  CheckSquare,
  Square
} from 'lucide-react';

interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'critical';
  category: 'trading' | 'system' | 'risk' | 'market' | 'portfolio' | 'security';
  source: 'prometheus' | 'trading-engine' | 'risk-manager' | 'market-data' | 'user-action' | 'system';
  timestamp: string;
  read: boolean;
  acknowledged: boolean;
  priority: 'low' | 'medium' | 'high' | 'critical';
  tags: string[];
  metadata?: {
    symbol?: string;
    value?: number;
    threshold?: number;
    metric?: string;
    instance?: string;
    severity?: string;
  };
}

interface AlertRule {
  id: number;
  name: string;
  description: string;
  category: string;
  enabled: boolean;
  conditions?: {
    metric: string;
    operator: '>' | '<' | '=' | '>=' | '<=';
    threshold: number;
    duration: string;
  };
  notifications: {
    email: boolean;
    sms: boolean;
    push: boolean;
    webhook: boolean;
  };
  severity: 'low' | 'medium' | 'high' | 'critical';
}

interface SystemMetric {
  name: string;
  value: number;
  unit: string;
  status: 'normal' | 'warning' | 'critical';
  timestamp: string;
  change: number;
}

// API utility functions for alert rules
async function fetchAlertRules(userId: number) {
  const res = await fetch(`/api/realtime/alerts?user_id=${userId}`);
  if (!res.ok) throw new Error('Failed to fetch alert rules');
  return res.json();
}

async function createAlertRule(userId: number, data: any) {
  const res = await fetch(`/api/realtime/alerts?user_id=${userId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to create alert rule');
  return res.json();
}

async function updateAlertRule(alertId: number, data: any) {
  const res = await fetch(`/api/realtime/alerts/${alertId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to update alert rule');
  return res.json();
}

async function deleteAlertRule(alertId: number) {
  const res = await fetch(`/api/realtime/alerts/${alertId}`, {
    method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete alert rule');
  return res.json();
}

// Format timestamp utility
function formatTimestamp(timestamp: string) {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return 'همین الان';
  if (diffMins < 60) return `${diffMins} دقیقه پیش`;
  if (diffHours < 24) return `${diffHours} ساعت پیش`;
  return `${diffDays} روز پیش`;
}

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [alertRules, setAlertRules] = useState<AlertRule[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetric[]>([]);
  const [selectedTab, setSelectedTab] = useState('alerts');
  const [filter, setFilter] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [showOnlyUnread, setShowOnlyUnread] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [notificationSettings, setNotificationSettings] = useState({
    email: true,
    sms: false,
    push: true,
    webhook: false
  });
  const [alertsLoading, setAlertsLoading] = useState(false);
  const [alertsError, setAlertsError] = useState<string | null>(null);
  const userId = 1; // TODO: Replace with real user ID

  // Fetch alert rules from backend
  useEffect(() => {
    setAlertsLoading(true);
    fetchAlertRules(userId)
      .then(setAlertRules)
      .catch((err) => setAlertsError(err.message))
      .finally(() => setAlertsLoading(false));
  }, []);

  // Create alert rule handler
  async function handleCreateAlertRule(data: any) {
    setAlertsLoading(true);
    try {
      const newRule = await createAlertRule(userId, data);
      setAlertRules((prev) => [...prev, newRule]);
    } catch (err: any) {
      setAlertsError(err.message);
    } finally {
      setAlertsLoading(false);
    }
  }

  // Enable/disable alert rule handler
  async function handleToggleAlertRule(rule: any) {
    setAlertsLoading(true);
    try {
      const updated = await updateAlertRule(Number(rule.id), { ...rule, enabled: !rule.enabled });
      setAlertRules((prev) => prev.map((r) => (Number(r.id) === Number(rule.id) ? updated : r)));
    } catch (err: any) {
      setAlertsError(err.message);
    } finally {
      setAlertsLoading(false);
    }
  }

  // Delete alert rule handler
  async function handleDeleteAlertRule(ruleId: number) {
    setAlertsLoading(true);
    try {
      await deleteAlertRule(ruleId);
      setAlertRules((prev) => prev.filter((r) => r.id !== ruleId));
    } catch (err: any) {
      setAlertsError(err.message);
    } finally {
      setAlertsLoading(false);
    }
  }

  const filteredNotifications = notifications.filter(notification => {
    const matchesFilter = filter === 'all' || notification.category === filter;
    const matchesSearch = notification.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         notification.message.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesReadFilter = !showOnlyUnread || !notification.read;

    return matchesFilter && matchesSearch && matchesReadFilter;
  });

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'success': return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'warning': return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'error': return <XCircle className="w-5 h-5 text-red-500" />;
      case 'critical': return <AlertOctagon className="w-5 h-5 text-red-600" />;
      default: return <Info className="w-5 h-5 text-blue-500" />;
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'trading': return <Activity className="w-4 h-4" />;
      case 'system': return <Server className="w-4 h-4" />;
      case 'risk': return <Shield className="w-4 h-4" />;
      case 'market': return <TrendingUp className="w-4 h-4" />;
      case 'portfolio': return <DollarSign className="w-4 h-4" />;
      case 'security': return <AlertCircle className="w-4 h-4" />;
      default: return <Bell className="w-4 h-4" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'success': return 'bg-green-100 text-green-800 border-green-200';
      case 'warning': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'error': return 'bg-red-100 text-red-800 border-red-200';
      case 'critical': return 'bg-red-200 text-red-900 border-red-300';
      default: return 'bg-blue-100 text-blue-800 border-blue-200';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-600';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const getMetricStatusColor = (status: string) => {
    switch (status) {
      case 'normal': return 'text-green-600';
      case 'warning': return 'text-yellow-600';
      case 'critical': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const markAsRead = (id: string) => {
    setNotifications(prev => prev.map(n =>
      n.id === id ? { ...n, read: true } : n
    ));
  };

  const markAsAcknowledged = (id: string) => {
    setNotifications(prev => prev.map(n =>
      n.id === id ? { ...n, acknowledged: true } : n
    ));
  };

  const deleteNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const unreadCount = notifications.filter(n => !n.read).length;
  const criticalCount = notifications.filter(n => n.type === 'critical' && !n.acknowledged).length;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <Bell className="w-8 h-8 text-blue-600" />
            اعلان‌ها و هشدارها
            {unreadCount > 0 && (
              <Badge className="bg-red-500 text-white">
                {unreadCount}
              </Badge>
            )}
          </h1>
          <p className="text-muted-foreground">
            سیستم پایش و مدیریت هشدار به‌صورت لحظه‌ای با یکپارچگی Prometheus
          </p>
        </div>

        {/* Quick Stats */}
        <div className="flex gap-4">
          <Card className="w-40">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-orange-600">{unreadCount}</div>
              <div className="text-xs text-gray-500">خوانده‌نشده</div>
            </CardContent>
          </Card>
          <Card className="w-40">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-red-600">{criticalCount}</div>
              <div className="text-xs text-gray-500">بحرانی</div>
            </CardContent>
          </Card>
          <Card className="w-40">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-green-600">{alertRules.filter(r => r.enabled).length}</div>
              <div className="text-xs text-gray-500">قوانین فعال</div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* System Metrics Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            معیارهای زنده سیستم
            <Badge variant="outline" className="ml-auto">
              <Activity className="w-3 h-3 mr-1" />
              Prometheus
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4 lg:grid-cols-8">
            {systemMetrics.map((metric, index) => (
              <div key={index} className="text-center">
                <div className="text-xs text-gray-500 mb-1">{metric.name}</div>
                <div className={`text-lg font-bold ${getMetricStatusColor(metric.status)}`}>
                  {typeof metric.value === 'number' ? metric.value.toLocaleString() : metric.value}
                  <span className="text-xs ml-1">{metric.unit}</span>
                </div>
                <div className={`text-xs ${metric.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {metric.change >= 0 ? '+' : ''}{metric.change.toFixed(1)}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="alerts" className="flex items-center gap-2">
            <Bell className="w-4 h-4" />
            هشدارها
            {unreadCount > 0 && (
              <Badge className="bg-red-500 text-white text-xs">
                {unreadCount}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="rules">قوانین</TabsTrigger>
          <TabsTrigger value="channels">کانال‌ها</TabsTrigger>
          <TabsTrigger value="settings">تنظیمات</TabsTrigger>
        </TabsList>

        <TabsContent value="alerts" className="space-y-4">
          {/* Filters and Search */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Filter className="w-5 h-5" />
                  فیلترها
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowOnlyUnread(!showOnlyUnread)}
                    className="flex items-center gap-2"
                  >
                    {showOnlyUnread ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                    {showOnlyUnread ? 'نمایش همه' : 'فقط خوانده‌نشده'}
                  </Button>
                  <Button variant="outline" size="sm">
                    <Archive className="w-4 h-4 mr-1" />
                    بایگانی خوانده‌شده‌ها
                  </Button>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-4">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                    <Input
                      placeholder="جستجوی اعلان‌ها..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>
                <select
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  className="px-3 py-2 border rounded-md"
                >
                  <option value="all">همه دسته‌ها</option>
                  <option value="trading">معاملات</option>
                  <option value="system">سیستم</option>
                  <option value="risk">ریسک</option>
                  <option value="market">بازار</option>
                  <option value="portfolio">پرتفوی</option>
                  <option value="security">امنیت</option>
                </select>
              </div>
            </CardContent>
          </Card>

          {/* Notifications List */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                تاریخچه هشدارها
              </CardTitle>
            </CardHeader>
            <CardContent>
              {filteredNotifications.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  اعلانی مطابق فیلترهای شما یافت نشد
                </div>
              ) : (
                <div className="space-y-3">
                  {filteredNotifications.map(notification => (
                    <div
                      key={notification.id}
                      className={`border rounded-lg p-4 ${
                        !notification.read ? 'bg-blue-50 border-blue-200' : 'bg-white'
                      } ${getTypeColor(notification.type)}`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3 flex-1">
                          <div className="flex items-center gap-2">
                            {getNotificationIcon(notification.type)}
                            <div className={`w-2 h-2 rounded-full ${getPriorityColor(notification.priority)}`} />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <h3 className="font-medium text-sm">{notification.title}</h3>
                              <Badge variant="outline" className="flex items-center gap-1 text-xs">
                                {getCategoryIcon(notification.category)}
                                {notification.category}
                              </Badge>
                              <Badge variant="secondary" className="text-xs">
                                {notification.source}
                              </Badge>
                            </div>
                            <p className="text-sm text-gray-600 mb-2">{notification.message}</p>

                            {notification.metadata && (
                              <div className="flex gap-4 text-xs text-gray-500 mb-2">
                                {notification.metadata.symbol && (
                                  <span>نماد: {notification.metadata.symbol}</span>
                                )}
                                {notification.metadata.value && (
                                  <span>مقدار: {notification.metadata.value.toLocaleString()}</span>
                                )}
                                {notification.metadata.threshold && (
                                  <span>آستانه: {notification.metadata.threshold.toLocaleString()}</span>
                                )}
                                {notification.metadata.instance && (
                                  <span>نمونه: {notification.metadata.instance}</span>
                                )}
                              </div>
                            )}

                            <div className="flex gap-2 flex-wrap">
                              {notification.tags.map(tag => (
                                <Badge key={tag} variant="outline" className="text-xs">
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-2 ml-4">
                          <div className="text-xs text-gray-500">
                            {formatTimestamp(notification.timestamp)}
                          </div>
                          <div className="flex gap-1">
                            {!notification.read && (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => markAsRead(notification.id)}
                              >
                                <CheckSquare className="w-3 h-3" />
                              </Button>
                            )}
                            {!notification.acknowledged && notification.type === 'critical' && (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => markAsAcknowledged(notification.id)}
                                className="text-orange-600"
                              >
                                <AlertTriangle className="w-3 h-3" />
                              </Button>
                            )}
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => deleteNotification(notification.id)}
                              className="text-red-600"
                            >
                              <Trash2 className="w-3 h-3" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="rules" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Settings className="w-5 h-5" />
                  قوانین هشدار
                </div>
                <Button className="flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  ایجاد قانون
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {alertRules.map(rule => (
                  <div
                    key={rule.id}
                    className={`border rounded-lg p-4 ${
                      rule.enabled ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2">
                          <div className={`w-3 h-3 rounded-full ${rule.enabled ? 'bg-green-500' : 'bg-gray-400'}`} />
                          <h3 className="font-medium">{rule.name}</h3>
                        </div>
                        <Badge variant="outline">{rule.category}</Badge>
                        <Badge className={`${
                          rule.severity === 'critical' ? 'bg-red-500' :
                          rule.severity === 'high' ? 'bg-orange-500' :
                          rule.severity === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                        } text-white`}>
                          {rule.severity}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleToggleAlertRule(rule)}
                        >
                          {rule.enabled ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                          {rule.enabled ? 'غیرفعال‌سازی' : 'فعال‌سازی'}
                        </Button>
                        <Button variant="outline" size="sm">
                          <Settings className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>

                    <p className="text-sm text-gray-600 mb-3">{rule.description}</p>

                    <div className="grid gap-4 md:grid-cols-2">
                      <div>
                        <h4 className="font-medium text-sm mb-2">شرایط</h4>
                        <div className="text-sm text-gray-600">
                          <code className="bg-gray-100 px-2 py-1 rounded text-xs">
                            {rule.conditions?.metric} {rule.conditions?.operator} {rule.conditions?.threshold}
                          </code>
                          <span className="ml-2">به مدت {rule.conditions?.duration}</span>
                        </div>
                      </div>

                      <div>
                        <h4 className="font-medium text-sm mb-2">کانال‌های اعلان</h4>
                        <div className="flex gap-2">
                          {rule.notifications.email && <Badge variant="outline"><Mail className="w-3 h-3 mr-1" />ایمیل</Badge>}
                          {rule.notifications.sms && <Badge variant="outline"><Smartphone className="w-3 h-3 mr-1" />پیامک</Badge>}
                          {rule.notifications.push && <Badge variant="outline"><Bell className="w-3 h-3 mr-1" />پوش</Badge>}
                          {rule.notifications.webhook && <Badge variant="outline"><Zap className="w-3 h-3 mr-1" />وبهوک</Badge>}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="channels" className="space-y-4">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Mail className="w-5 h-5" />
                  اعلان‌های ایمیلی
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>آدرس ایمیل</Label>
                  <Input placeholder="alerts@yourcompany.com" />
                </div>
                <div className="space-y-2">
                  <Label>انواع اعلان</Label>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked />
                      <span className="text-sm">هشدارهای بحرانی</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked />
                      <span className="text-sm">هشدارهای سیستم</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" />
                      <span className="text-sm">اعلان‌های معاملاتی</span>
                    </label>
                  </div>
                </div>
                <Button className="w-full">به‌روزرسانی تنظیمات ایمیل</Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Smartphone className="w-5 h-5" />
                  اعلان‌های پیامکی
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>شماره تلفن</Label>
                  <Input placeholder="+1 (555) 123-4567" />
                </div>
                <div className="space-y-2">
                  <Label>فقط موارد اضطراری</Label>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked />
                      <span className="text-sm">خرابی‌های بحرانی سیستم</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked />
                      <span className="text-sm">نقض محدودیت ریسک</span>
                    </label>
                  </div>
                </div>
                <Button className="w-full">به‌روزرسانی تنظیمات پیامک</Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  یکپارچگی وبهوک
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>آدرس وبهوک</Label>
                  <Input placeholder="https://your-system.com/webhooks/alerts" />
                </div>
                <div className="space-y-2">
                  <Label>توکن محرمانه</Label>
                  <Input type="password" placeholder="webhook-secret-token" />
                </div>
                <div className="space-y-2">
                  <Label>قالب Payload</Label>
                  <select className="w-full p-2 border rounded-md">
                    <option>JSON</option>
                    <option>Slack</option>
                    <option>Discord</option>
                    <option>Teams</option>
                  </select>
                </div>
                <Button className="w-full">تست وبهوک</Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="w-5 h-5" />
                  یکپارچگی Prometheus
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>آدرس Prometheus</Label>
                  <Input value="http://prometheus:9090" readOnly />
                </div>
                <div className="space-y-2">
                  <Label>آدرس AlertManager</Label>
                  <Input value="http://alertmanager:9093" readOnly />
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" className="flex-1">
                    <RefreshCw className="w-4 h-4 mr-1" />
                    همگام‌سازی قوانین
                  </Button>
                  <Button variant="outline" className="flex-1">
                    <Download className="w-4 h-4 mr-1" />
                    خروجی تنظیمات
                  </Button>
                </div>
                <div className="text-sm text-green-600 dark:text-green-400 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4" />
                  متصل — دریافت داده از API، معاملات، ریسک، دیتابیس، Redis، Celery و معیارهای سیستم
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Volume2 className="w-5 h-5" />
                  صدا و نمایش
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">اعلان‌های صوتی</div>
                    <div className="text-sm text-gray-500">پخش صدا برای هشدارهای جدید</div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSoundEnabled(!soundEnabled)}
                  >
                    {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
                  </Button>
                </div>

                <div className="space-y-2">
                  <Label>صدای هشدار</Label>
                  <select className="w-full p-2 border rounded-md">
                    <option>زنگ پیش‌فرض</option>
                    <option>بوق فوری</option>
                    <option>زنگ ملایم</option>
                    <option>زنگ معاملاتی</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <Label>اعلان‌های دسکتاپ</Label>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked />
                      <span className="text-sm">نمایش اعلان‌های مرورگر</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked />
                      <span className="text-sm">نمایش تعداد روی تب</span>
                    </label>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="w-5 h-5" />
                  زمان‌بندی و تناوب
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>ساعات سکوت</Label>
                  <div className="grid grid-cols-2 gap-2">
                    <Input type="time" defaultValue="22:00" />
                    <Input type="time" defaultValue="08:00" />
                  </div>
                  <div className="text-xs text-gray-500">
                    فقط هشدارهای بحرانی در ساعات سکوت
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>گروه‌بندی اعلان‌ها</Label>
                  <select className="w-full p-2 border rounded-md">
                    <option>لحظه‌ای</option>
                    <option>هر ۵ دقیقه</option>
                    <option>هر ۱۵ دقیقه</option>
                    <option>خلاصه ساعتی</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <Label>تأیید خودکار</Label>
                  <select className="w-full p-2 border rounded-md">
                    <option>هرگز</option>
                    <option>بعد از ۱ ساعت</option>
                    <option>بعد از ۲۴ ساعت</option>
                    <option>بعد از ۷ روز</option>
                  </select>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  تشدید هشدار
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <Label className="text-sm font-medium">سطح ۱ - اصلی</Label>
                    <Input placeholder="primary-oncall@company.com" className="mt-1" />
                  </div>
                  <div>
                    <Label className="text-sm font-medium">سطح ۲ - ثانویه (۱۵ دقیقه)</Label>
                    <Input placeholder="secondary-oncall@company.com" className="mt-1" />
                  </div>
                  <div>
                    <Label className="text-sm font-medium">سطح ۳ - مدیر (۳۰ دقیقه)</Label>
                    <Input placeholder="manager@company.com" className="mt-1" />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="flex items-center gap-2">
                    <input type="checkbox" defaultChecked />
                    <span className="text-sm">فعال‌سازی تشدید برای هشدارهای بحرانی</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input type="checkbox" />
                    <span className="text-sm">افزودن جزئیات در تشدید</span>
                  </label>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <User className="w-5 h-5" />
                  تنظیمات شخصی
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>منطقه زمانی</Label>
                  <select className="w-full p-2 border rounded-md">
                    <option>UTC-8 (وقت اقیانوس آرام)</option>
                    <option>UTC-5 (وقت شرقی)</option>
                    <option>UTC+0 (GMT)</option>
                    <option>UTC+1 (CET)</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <Label>زبان</Label>
                  <select className="w-full p-2 border rounded-md">
                    <option>انگلیسی</option>
                    <option>اسپانیایی</option>
                    <option>فرانسوی</option>
                    <option>آلمانی</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <Label>قالب تاریخ</Label>
                  <select className="w-full p-2 border rounded-md">
                    <option>MM/DD/YYYY</option>
                    <option>DD/MM/YYYY</option>
                    <option>YYYY-MM-DD</option>
                  </select>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}