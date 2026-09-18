'use client';

import { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Shield,
  Users,
  Settings,
  Monitor,
  Database,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  User,
  UserCheck,
  UserX,
  Search,
  Filter,
  Edit,
  Trash2,
  Plus,
  Download,
  Upload,
  RefreshCw,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  Calendar,
  Clock,
  BarChart3,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Zap,
  Server,
  Cpu,
  HardDrive,
  Network,
  Globe,
  Key,
  Mail,
  Phone,
  MapPin,
  Building,
  Briefcase,
  CreditCard,
  Archive,
  FileText,
  LogOut,
  Bell,
  Wrench,
  Target,
  AlertCircle,
  CheckSquare,
  Square,
  MoreHorizontal
} from 'lucide-react';
import { StartupTrackerPanel } from '@/components/admin/startup-tracker-panel';

interface AdminUser {
  id: number;
  name: string;
  email: string;
  phone: string | null;
  role: string;
  is_active: boolean;
  risk_tolerance: string;
  permissions: string[];
  last_login: string | null;
  created_at: string | null;
  total_trades: number;
  portfolio_value: number;
}

interface SystemHealth {
  service: string;
  status: 'healthy' | 'warning' | 'critical' | 'offline';
  uptime: string;
  cpu: number;
  memory: number;
  responseTime: number;
  lastCheck: string;
  version: string;
}

interface AuditLog {
  id: number;
  timestamp: string;
  actor: string;
  action: string;
  target_type: string | null;
  target_id: string | null;
  detail: Record<string, unknown> | null;
  ip_address: string | null;
}

interface ConfigSetting {
  id: string;
  category: string;
  key: string;
  value: string;
  type: 'string' | 'number' | 'boolean' | 'json';
  description: string;
  modified: string;
  modifiedBy: string;
  requiresRestart: boolean;
}

const AuditLogPage = dynamic(
  () => import('@/app/audit-log/page').then((m) => m.default),
  { ssr: false, loading: () => <div className="p-6 text-muted-foreground">در حال بارگذاری لاگ ممیزی…</div> }
);

const STATUS_LABELS: Record<string, string> = {
  healthy: 'سالم',
  warning: 'هشدار',
  critical: 'بحرانی',
  offline: 'آفلاین',
  active: 'فعال',
  inactive: 'غیرفعال',
  suspended: 'معلق',
  pending: 'در انتظار',
  success: 'موفق',
  failure: 'ناموفق',
};

const ROLE_LABELS: Record<string, string> = {
  admin: 'مدیر',
  trader: 'معامله‌گر',
  analyst: 'تحلیل‌گر',
  viewer: 'بیننده',
};

export default function AdminPage() {
  const [selectedTab, setSelectedTab] = useState('overview');
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [systemHealth, setSystemHealth] = useState<SystemHealth[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [configSettings, setConfigSettings] = useState<ConfigSetting[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRole, setSelectedRole] = useState('all');
  const [selectedStatus, setSelectedStatus] = useState('all');

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [pendingUserId, setPendingUserId] = useState<number | null>(null);

  // داده‌های واقعی از بک‌اند (issue #12) — دیگر mock نیست.
  const loadUsers = useCallback(async () => {
    const res = await fetch('/api/admin/users', { cache: 'no-store' });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? 'خطا در دریافت کاربران');
    setUsers(await res.json());
  }, []);

  const loadAuditLogs = useCallback(async () => {
    const res = await fetch('/api/admin/audit-log?limit=100', { cache: 'no-store' });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? 'خطا در دریافت لاگ');
    setAuditLogs(await res.json());
  }, []);

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        await Promise.all([loadUsers(), loadAuditLogs()]);
        setError('');
      } catch (e) {
        setError(e instanceof Error ? e.message : 'خطا در بارگذاری داده‌ها');
      } finally {
        setLoading(false);
      }
    })();
  }, [loadUsers, loadAuditLogs]);

  const updateUser = async (userId: number, patch: { role?: string; is_active?: boolean }) => {
    setPendingUserId(userId);
    try {
      const res = await fetch(`/api/admin/users/${userId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? 'خطا در بروزرسانی کاربر');
      setUsers(prev => prev.map(u => (u.id === userId ? { ...u, ...patch } as AdminUser : u)));
      await loadAuditLogs();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'خطا در بروزرسانی کاربر');
    } finally {
      setPendingUserId(null);
    }
  };

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRole = selectedRole === 'all' || user.role === selectedRole;
    const matchesStatus = selectedStatus === 'all'
      || (selectedStatus === 'active' ? user.is_active : !user.is_active);
    return matchesSearch && matchesRole && matchesStatus;
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': case 'active': return 'bg-green-100 text-green-800';
      case 'warning': case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'critical': case 'suspended': return 'bg-red-100 text-red-800';
      case 'offline': case 'inactive': return 'bg-gray-100 text-gray-800';
      default: return 'bg-blue-100 text-blue-800';
    }
  };

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'admin': return 'bg-purple-100 text-purple-800';
      case 'trader': return 'bg-blue-100 text-blue-800';
      case 'analyst': return 'bg-green-100 text-green-800';
      case 'viewer': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      notation: value >= 1000000 ? 'compact' : 'standard',
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('fa-IR');
  };

  const totalUsers = users.length;
  const activeUsers = users.filter(u => u.is_active).length;
  const adminCount = users.filter(u => u.role === 'admin').length;
  const inactiveUsers = users.filter(u => !u.is_active).length;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <Shield className="w-8 h-8 text-purple-600" />
            پنل مدیریت
          </h1>
          <p className="text-muted-foreground">
            مدیریت سیستم، مدیریت کاربران و پیکربندی پلتفرم
          </p>
        </div>

        {/* Quick Stats */}
        <div className="flex gap-4">
          <Card className="w-32">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-blue-600">{totalUsers}</div>
              <div className="text-xs text-gray-500">کل کاربران</div>
            </CardContent>
          </Card>
          <Card className="w-32">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-green-600">{activeUsers}</div>
              <div className="text-xs text-gray-500">فعال</div>
            </CardContent>
          </Card>
          <Card className="w-32">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-purple-600">{adminCount}</div>
              <div className="text-xs text-gray-500">مدیران</div>
            </CardContent>
          </Card>
          <Card className="w-32">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-red-600">{inactiveUsers}</div>
              <div className="text-xs text-gray-500">غیرفعال</div>
            </CardContent>
          </Card>
        </div>
      </div>

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-7">
          <TabsTrigger value="overview">نمای کلی</TabsTrigger>
          <TabsTrigger value="users">کاربران</TabsTrigger>
          <TabsTrigger value="system">سلامت سیستم</TabsTrigger>
          <TabsTrigger value="audit">لاگ‌های ممیزی</TabsTrigger>
          <TabsTrigger value="config">پیکربندی</TabsTrigger>
          <TabsTrigger value="tools">ابزارها</TabsTrigger>
          <TabsTrigger value="startup">استارتاپ‌تراکر</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {error && (
            <Card>
              <CardContent className="p-4 text-red-600 text-sm">{error}</CardContent>
            </Card>
          )}

          {/* User Status Overview */}
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="w-5 h-5" />
                  ترکیب نقش‌ها
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {loading ? (
                    <div className="text-sm text-muted-foreground">در حال بارگذاری…</div>
                  ) : users.length === 0 ? (
                    <div className="text-sm text-muted-foreground">کاربری ثبت نشده است</div>
                  ) : (
                    Object.entries(
                      users.reduce<Record<string, number>>((acc, u) => {
                        acc[u.role] = (acc[u.role] ?? 0) + 1;
                        return acc;
                      }, {})
                    ).map(([role, count]) => (
                      <div key={role} className="flex items-center justify-between p-2 border rounded">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-sm">{ROLE_LABELS[role] ?? role}</span>
                        </div>
                        <Badge className={getRoleColor(role)}>{count}</Badge>
                      </div>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="w-5 h-5" />
                  فعالیت کاربران
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">کاربران فعال</span>
                    <span className="font-medium">{activeUsers} / {totalUsers}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">غیرفعال</span>
                    <span className="font-medium">{inactiveUsers}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">مدیران</span>
                    <span className="font-medium">{adminCount}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">کل معاملات ثبت‌شده</span>
                    <span className="font-medium">
                      {users.reduce((sum, u) => sum + u.total_trades, 0).toLocaleString()}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recent Activity */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                فعالیت‌های مدیریتی اخیر
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {loading ? (
                  <div className="text-sm text-muted-foreground">در حال بارگذاری…</div>
                ) : auditLogs.length === 0 ? (
                  <div className="text-sm text-muted-foreground">رویدادی ثبت نشده است</div>
                ) : (
                  auditLogs.slice(0, 5).map(log => (
                    <div key={log.id} className="flex items-center justify-between p-3 border rounded">
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 rounded-full bg-blue-500" />
                        <div>
                          <div className="font-medium text-sm">{log.action.replace(/_/g, ' ')}</div>
                          <div className="text-xs text-gray-500">
                            {log.actor} • {formatTimestamp(log.timestamp)}
                          </div>
                        </div>
                      </div>
                      {log.target_type && (
                        <Badge variant="outline">
                          {log.target_type}{log.target_id ? ` #${log.target_id}` : ''}
                        </Badge>
                      )}
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="users" className="space-y-4">
          {/* User Management Controls */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Users className="w-5 h-5" />
                  مدیریت کاربران
                </div>
                <Button className="flex items-center gap-2">
                  <Plus className="w-4 h-4" />
                  افزودن کاربر
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-4">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                    <Input
                      placeholder="جستجوی کاربران..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>
                <select
                  value={selectedRole}
                  onChange={(e) => setSelectedRole(e.target.value)}
                  className="px-3 py-2 border rounded-md"
                >
                  <option value="all">همه نقش‌ها</option>
                  <option value="admin">مدیر</option>
                  <option value="trader">معامله‌گر</option>
                  <option value="analyst">تحلیل‌گر</option>
                  <option value="viewer">بیننده</option>
                </select>
                <select
                  value={selectedStatus}
                  onChange={(e) => setSelectedStatus(e.target.value)}
                  className="px-3 py-2 border rounded-md"
                >
                  <option value="all">همه وضعیت‌ها</option>
                  <option value="active">فعال</option>
                  <option value="inactive">غیرفعال</option>
                  <option value="suspended">معلق</option>
                  <option value="pending">در انتظار</option>
                </select>
              </div>
            </CardContent>
          </Card>

          {/* Users Table */}
          <Card>
            <CardContent className="p-0">
              <div className="space-y-3 p-6">
                {loading ? (
                  <div className="text-sm text-muted-foreground">در حال بارگذاری…</div>
                ) : filteredUsers.length === 0 ? (
                  <div className="text-sm text-muted-foreground">کاربری با این فیلترها پیدا نشد</div>
                ) : (
                  filteredUsers.map(user => (
                    <div key={user.id} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                            <User className="w-5 h-5 text-blue-600" />
                          </div>
                          <div>
                            <div className="font-medium">{user.name}</div>
                            <div className="text-sm text-gray-500">{user.email}</div>
                          </div>
                          <Badge className={getRoleColor(user.role)}>
                            {ROLE_LABELS[user.role] ?? user.role}
                          </Badge>
                          <Badge className={getStatusColor(user.is_active ? 'active' : 'inactive')}>
                            {user.is_active ? 'فعال' : 'غیرفعال'}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2">
                          <select
                            value={user.role}
                            disabled={pendingUserId === user.id}
                            onChange={(e) => updateUser(user.id, { role: e.target.value })}
                            className="px-2 py-1 border rounded-md text-sm disabled:opacity-50"
                          >
                            <option value="admin">مدیر</option>
                            <option value="trader">معامله‌گر</option>
                            <option value="demo">دمو</option>
                          </select>
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={pendingUserId === user.id}
                            onClick={() => updateUser(user.id, { is_active: !user.is_active })}
                          >
                            {user.is_active ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />}
                          </Button>
                        </div>
                      </div>

                      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 text-sm">
                        <div>
                          <div className="text-gray-500">ریسک‌پذیری</div>
                          <div className="font-medium">{user.risk_tolerance}</div>
                        </div>
                        <div>
                          <div className="text-gray-500">کل معاملات</div>
                          <div className="font-medium">{user.total_trades.toLocaleString()}</div>
                        </div>
                        <div>
                          <div className="text-gray-500">ارزش پرتفوی</div>
                          <div className="font-medium">{formatCurrency(user.portfolio_value)}</div>
                        </div>
                        <div>
                          <div className="text-gray-500">آخرین ورود</div>
                          <div className="font-medium">
                            {user.last_login ? formatTimestamp(user.last_login) : '—'}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="system" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Monitor className="w-5 h-5" />
                  پایش سلامت سیستم
                </div>
                <Button variant="outline" size="sm">
                  <RefreshCw className="w-4 h-4 mr-1" />
                  بازخوانی
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {systemHealth.map(service => (
                  <div key={service.service} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className={`w-3 h-3 rounded-full ${
                          service.status === 'healthy' ? 'bg-green-500' :
                          service.status === 'warning' ? 'bg-yellow-500' :
                          service.status === 'critical' ? 'bg-red-500' : 'bg-gray-500'
                        }`} />
                        <h3 className="font-medium">{service.service}</h3>
                        <Badge className={getStatusColor(service.status)}>
                          {STATUS_LABELS[service.status] ?? service.status}
                        </Badge>
                        <Badge variant="outline">{service.version}</Badge>
                      </div>
                      <div className="text-sm text-gray-500">
                        آخرین بررسی: {formatTimestamp(service.lastCheck)}
                      </div>
                    </div>

                    <div className="grid gap-4 md:grid-cols-5 text-sm">
                      <div>
                        <div className="text-gray-500">زمان فعالیت</div>
                        <div className="font-medium">{service.uptime}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">مصرف CPU</div>
                        <div className="font-medium">{service.cpu}%</div>
                      </div>
                      <div>
                        <div className="text-gray-500">حافظه</div>
                        <div className="font-medium">{service.memory}%</div>
                      </div>
                      <div>
                        <div className="text-gray-500">زمان پاسخ</div>
                        <div className="font-medium">{service.responseTime}ms</div>
                      </div>
                      <div>
                        <div className="text-gray-500">عملیات</div>
                        <div className="flex gap-1">
                          <Button variant="outline" size="sm">
                            <Eye className="w-3 h-3" />
                          </Button>
                          <Button variant="outline" size="sm">
                            <Settings className="w-3 h-3" />
                          </Button>
                          <Button variant="outline" size="sm">
                            <RefreshCw className="w-3 h-3" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="audit" className="space-y-4">
          <AuditLogPage />
        </TabsContent>

        <TabsContent value="config" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                پیکربندی سیستم
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {configSettings.map(setting => (
                  <div key={setting.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <span className="font-medium">{setting.key}</span>
                        <Badge variant="outline">{setting.category}</Badge>
                        <Badge variant="outline">{setting.type}</Badge>
                        {setting.requiresRestart && (
                          <Badge className="bg-orange-100 text-orange-800">
                            نیاز به راه‌اندازی مجدد
                          </Badge>
                        )}
                      </div>
                      <Button variant="outline" size="sm">
                        <Edit className="w-4 h-4" />
                      </Button>
                    </div>
                    <div className="text-sm text-gray-600 mb-2">{setting.description}</div>
                    <div className="flex items-center gap-4 text-sm">
                      <span className="font-mono bg-gray-100 px-2 py-1 rounded">
                        {setting.value}
                      </span>
                      <span className="text-gray-500">
                        تغییر در {formatTimestamp(setting.modified)} توسط {setting.modifiedBy}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tools" className="space-y-4">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="w-5 h-5" />
                  ابزارهای پایگاه‌داده
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full justify-start">
                  <Download className="w-4 h-4 mr-2" />
                  خروجی پشتیبان پایگاه‌داده
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Upload className="w-4 h-4 mr-2" />
                  وارد کردن داده
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Wrench className="w-4 h-4 mr-2" />
                  نگه‌داری پایگاه‌داده
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <BarChart3 className="w-4 h-4 mr-2" />
                  تحلیل عملکرد
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  ابزارهای امنیتی
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full justify-start">
                  <Key className="w-4 h-4 mr-2" />
                  چرخش کلیدهای API
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Lock className="w-4 h-4 mr-2" />
                  اجبار بازنشانی رمز عبور
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <AlertTriangle className="w-4 h-4 mr-2" />
                  اسکن امنیتی
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <FileText className="w-4 h-4 mr-2" />
                  گزارش امنیتی
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Monitor className="w-5 h-5" />
                  ابزارهای سیستم
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full justify-start">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  راه‌اندازی مجدد سرویس‌ها
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Activity className="w-4 h-4 mr-2" />
                  بررسی سلامت
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Archive className="w-4 h-4 mr-2" />
                  آرشیو لاگ‌های قدیمی
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Target className="w-4 h-4 mr-2" />
                  تست عملکرد
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bell className="w-5 h-5" />
                  ابزارهای اعلان
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full justify-start">
                  <Mail className="w-4 h-4 mr-2" />
                  ارسال اعلان سیستمی
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <AlertCircle className="w-4 h-4 mr-2" />
                  هشدار اضطراری
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Bell className="w-4 h-4 mr-2" />
                  آزمایش اعلان‌ها
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Settings className="w-4 h-4 mr-2" />
                  تنظیمات هشدار
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="startup" className="space-y-4">
          <StartupTrackerPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
