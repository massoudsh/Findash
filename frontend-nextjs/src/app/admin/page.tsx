'use client';

import { useState, useEffect } from 'react';
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
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'trader' | 'analyst' | 'viewer';
  status: 'active' | 'inactive' | 'suspended' | 'pending';
  lastLogin: string;
  createdAt: string;
  totalTrades: number;
  portfolioValue: number;
  riskLevel: 'low' | 'medium' | 'high';
  permissions: string[];
  location: string;
  department: string;
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
  id: string;
  timestamp: string;
  user: string;
  action: string;
  resource: string;
  details: string;
  ipAddress: string;
  result: 'success' | 'failure';
  severity: 'low' | 'medium' | 'high' | 'critical';
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

  // Sample data initialization
  useEffect(() => {
    const sampleUsers: AdminUser[] = [
      {
        id: 'user-1',
        name: 'علی احمدی',
        email: 'ali.ahmadi@company.com',
        role: 'admin',
        status: 'active',
        lastLogin: '2024-01-20T14:30:00Z',
        createdAt: '2024-01-01T00:00:00Z',
        totalTrades: 1247,
        portfolioValue: 2500000,
        riskLevel: 'medium',
        permissions: ['USER_MANAGEMENT', 'SYSTEM_CONFIG', 'AUDIT_LOGS', 'TRADING_OVERRIDE'],
        location: 'تهران، ایران',
        department: 'عملیات معاملاتی'
      },
      {
        id: 'user-2',
        name: 'سارا کریمی',
        email: 'sara.karimi@company.com',
        role: 'trader',
        status: 'active',
        lastLogin: '2024-01-20T15:45:00Z',
        createdAt: '2024-01-05T00:00:00Z',
        totalTrades: 856,
        portfolioValue: 1800000,
        riskLevel: 'high',
        permissions: ['PORTFOLIO_MANAGEMENT', 'ORDER_EXECUTION', 'RISK_OVERRIDE'],
        location: 'لندن، انگلستان',
        department: 'معاملات سهام'
      },
      {
        id: 'user-3',
        name: 'محمد رضایی',
        email: 'mohammad.rezaei@company.com',
        role: 'analyst',
        status: 'active',
        lastLogin: '2024-01-20T12:15:00Z',
        createdAt: '2024-01-10T00:00:00Z',
        totalTrades: 234,
        portfolioValue: 750000,
        riskLevel: 'low',
        permissions: ['ANALYTICS_ACCESS', 'REPORT_GENERATION', 'DATA_EXPORT'],
        location: 'سنگاپور',
        department: 'پژوهش و تحلیل'
      },
      {
        id: 'user-4',
        name: 'مریم داوودی',
        email: 'maryam.davoodi@company.com',
        role: 'viewer',
        status: 'suspended',
        lastLogin: '2024-01-18T09:30:00Z',
        createdAt: '2024-01-15T00:00:00Z',
        totalTrades: 0,
        portfolioValue: 0,
        riskLevel: 'low',
        permissions: ['READ_ONLY_ACCESS'],
        location: 'تورنتو، کانادا',
        department: 'انطباق'
      }
    ];

    const sampleSystemHealth: SystemHealth[] = [
      {
        service: 'موتور معاملات',
        status: 'healthy',
        uptime: '99.98%',
        cpu: 45.2,
        memory: 68.7,
        responseTime: 12,
        lastCheck: '2024-01-20T16:00:00Z',
        version: 'v2.4.1'
      },
      {
        service: 'فید داده بازار',
        status: 'healthy',
        uptime: '99.95%',
        cpu: 32.1,
        memory: 54.3,
        responseTime: 8,
        lastCheck: '2024-01-20T16:00:00Z',
        version: 'v1.8.3'
      },
      {
        service: 'مدیریت ریسک',
        status: 'warning',
        uptime: '99.12%',
        cpu: 78.9,
        memory: 82.1,
        responseTime: 45,
        lastCheck: '2024-01-20T16:00:00Z',
        version: 'v3.1.0'
      },
      {
        service: 'خوشه پایگاه‌داده',
        status: 'healthy',
        uptime: '99.99%',
        cpu: 23.4,
        memory: 45.8,
        responseTime: 15,
        lastCheck: '2024-01-20T16:00:00Z',
        version: 'PostgreSQL 15.2'
      },
      {
        service: 'دروازه API',
        status: 'healthy',
        uptime: '99.87%',
        cpu: 38.7,
        memory: 62.3,
        responseTime: 22,
        lastCheck: '2024-01-20T16:00:00Z',
        version: 'v1.2.8'
      },
      {
        service: 'موتور تحلیل',
        status: 'critical',
        uptime: '87.23%',
        cpu: 95.4,
        memory: 98.2,
        responseTime: 156,
        lastCheck: '2024-01-20T16:00:00Z',
        version: 'v4.0.1'
      }
    ];

    const sampleAuditLogs: AuditLog[] = [
      {
        id: 'log-1',
        timestamp: '2024-01-20T15:45:23Z',
        user: 'ali.ahmadi@company.com',
        action: 'USER_SUSPENDED',
        resource: 'users/maryam.davoodi',
        details: 'کاربر به دلیل نقض سیاست‌ها معلق شد',
        ipAddress: '192.168.1.145',
        result: 'success',
        severity: 'high'
      },
      {
        id: 'log-2',
        timestamp: '2024-01-20T15:30:12Z',
        user: 'sara.karimi@company.com',
        action: 'LARGE_ORDER_EXECUTED',
        resource: 'orders/ord-789456',
        details: 'سفارش خرید ۵۰,۰۰۰ سهم AAPL اجرا شد',
        ipAddress: '10.0.0.234',
        result: 'success',
        severity: 'medium'
      },
      {
        id: 'log-3',
        timestamp: '2024-01-20T15:15:45Z',
        user: 'system',
        action: 'CONFIG_CHANGED',
        resource: 'settings/risk_limits',
        details: 'حد VaR پرتفوی از ۴۵ هزار دلار به ۵۰ هزار دلار تغییر یافت',
        ipAddress: '127.0.0.1',
        result: 'success',
        severity: 'medium'
      },
      {
        id: 'log-4',
        timestamp: '2024-01-20T14:58:33Z',
        user: 'mohammad.rezaei@company.com',
        action: 'LOGIN_FAILED',
        resource: 'auth/login',
        details: 'تلاش ناموفق برای ورود - اطلاعات نامعتبر',
        ipAddress: '203.45.67.89',
        result: 'failure',
        severity: 'low'
      }
    ];

    const sampleConfigSettings: ConfigSetting[] = [
      {
        id: 'config-1',
        category: 'Trading',
        key: 'max_order_size',
        value: '1000000',
        type: 'number',
        description: 'حداکثر حجم سفارش به دلار',
        modified: '2024-01-20T10:30:00Z',
        modifiedBy: 'ali.ahmadi@company.com',
        requiresRestart: false
      },
      {
        id: 'config-2',
        category: 'Risk',
        key: 'portfolio_var_limit',
        value: '50000',
        type: 'number',
        description: 'حد ارزش در معرض ریسک پرتفوی به دلار',
        modified: '2024-01-20T15:15:00Z',
        modifiedBy: 'system',
        requiresRestart: false
      },
      {
        id: 'config-3',
        category: 'System',
        key: 'session_timeout',
        value: '3600',
        type: 'number',
        description: 'مهلت زمانی نشست کاربر به ثانیه',
        modified: '2024-01-19T14:20:00Z',
        modifiedBy: 'ali.ahmadi@company.com',
        requiresRestart: true
      },
      {
        id: 'config-4',
        category: 'Security',
        key: 'enable_2fa',
        value: 'true',
        type: 'boolean',
        description: 'الزام تأیید دومرحله‌ای',
        modified: '2024-01-18T09:45:00Z',
        modifiedBy: 'security-admin@company.com',
        requiresRestart: false
      }
    ];

    setUsers(sampleUsers);
    setSystemHealth(sampleSystemHealth);
    setAuditLogs(sampleAuditLogs);
    setConfigSettings(sampleConfigSettings);
  }, []);

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRole = selectedRole === 'all' || user.role === selectedRole;
    const matchesStatus = selectedStatus === 'all' || user.status === selectedStatus;
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
  const activeUsers = users.filter(u => u.status === 'active').length;
  const healthyServices = systemHealth.filter(s => s.status === 'healthy').length;
  const criticalIssues = systemHealth.filter(s => s.status === 'critical').length;

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
              <div className="text-2xl font-bold text-green-600">{healthyServices}</div>
              <div className="text-xs text-gray-500">سرویس‌های سالم</div>
            </CardContent>
          </Card>
          <Card className="w-32">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-red-600">{criticalIssues}</div>
              <div className="text-xs text-gray-500">مشکلات بحرانی</div>
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
          {/* System Status Overview */}
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Monitor className="w-5 h-5" />
                  وضعیت سیستم
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {systemHealth.slice(0, 4).map(service => (
                    <div key={service.service} className="flex items-center justify-between p-2 border rounded">
                      <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${
                          service.status === 'healthy' ? 'bg-green-500' :
                          service.status === 'warning' ? 'bg-yellow-500' :
                          service.status === 'critical' ? 'bg-red-500' : 'bg-gray-500'
                        }`} />
                        <span className="font-medium text-sm">{service.service}</span>
                      </div>
                      <Badge className={getStatusColor(service.status)}>
                        {STATUS_LABELS[service.status] ?? service.status}
                      </Badge>
                    </div>
                  ))}
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
                    <span className="text-sm text-gray-600">معلق</span>
                    <span className="font-medium">{users.filter(u => u.status === 'suspended').length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">در انتظار</span>
                    <span className="font-medium">{users.filter(u => u.status === 'pending').length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">کل معاملات امروز</span>
                    <span className="font-medium">۲,۳۳۷</span>
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
                {auditLogs.slice(0, 5).map(log => (
                  <div key={log.id} className="flex items-center justify-between p-3 border rounded">
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${
                        log.severity === 'critical' ? 'bg-red-500' :
                        log.severity === 'high' ? 'bg-orange-500' :
                        log.severity === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                      }`} />
                      <div>
                        <div className="font-medium text-sm">{log.action.replace('_', ' ')}</div>
                        <div className="text-xs text-gray-500">{log.user} • {formatTimestamp(log.timestamp)}</div>
                      </div>
                    </div>
                    <Badge className={log.result === 'success' ? getStatusColor('healthy') : getStatusColor('critical')}>
                      {STATUS_LABELS[log.result] ?? log.result}
                    </Badge>
                  </div>
                ))}
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
                {filteredUsers.map(user => (
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
                        <Badge className={getStatusColor(user.status)}>
                          {STATUS_LABELS[user.status] ?? user.status}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button variant="outline" size="sm">
                          <Edit className="w-4 h-4" />
                        </Button>
                        <Button variant="outline" size="sm">
                          {user.status === 'active' ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />}
                        </Button>
                        <Button variant="outline" size="sm" className="text-red-600">
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>

                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5 text-sm">
                      <div>
                        <div className="text-gray-500">واحد</div>
                        <div className="font-medium">{user.department}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">موقعیت</div>
                        <div className="font-medium">{user.location}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">کل معاملات</div>
                        <div className="font-medium">{user.totalTrades.toLocaleString()}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">ارزش پرتفوی</div>
                        <div className="font-medium">{formatCurrency(user.portfolioValue)}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">آخرین ورود</div>
                        <div className="font-medium">{formatTimestamp(user.lastLogin)}</div>
                      </div>
                    </div>
                  </div>
                ))}
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
