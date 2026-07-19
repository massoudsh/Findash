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
        name: 'John Anderson',
        email: 'john.anderson@company.com',
        role: 'admin',
        status: 'active',
        lastLogin: '2024-01-20T14:30:00Z',
        createdAt: '2024-01-01T00:00:00Z',
        totalTrades: 1247,
        portfolioValue: 2500000,
        riskLevel: 'medium',
        permissions: ['USER_MANAGEMENT', 'SYSTEM_CONFIG', 'AUDIT_LOGS', 'TRADING_OVERRIDE'],
        location: 'New York, NY',
        department: 'Trading Operations'
      },
      {
        id: 'user-2',
        name: 'Sarah Chen',
        email: 'sarah.chen@company.com',
        role: 'trader',
        status: 'active',
        lastLogin: '2024-01-20T15:45:00Z',
        createdAt: '2024-01-05T00:00:00Z',
        totalTrades: 856,
        portfolioValue: 1800000,
        riskLevel: 'high',
        permissions: ['PORTFOLIO_MANAGEMENT', 'ORDER_EXECUTION', 'RISK_OVERRIDE'],
        location: 'London, UK',
        department: 'Equity Trading'
      },
      {
        id: 'user-3',
        name: 'Michael Rodriguez',
        email: 'michael.rodriguez@company.com',
        role: 'analyst',
        status: 'active',
        lastLogin: '2024-01-20T12:15:00Z',
        createdAt: '2024-01-10T00:00:00Z',
        totalTrades: 234,
        portfolioValue: 750000,
        riskLevel: 'low',
        permissions: ['ANALYTICS_ACCESS', 'REPORT_GENERATION', 'DATA_EXPORT'],
        location: 'Singapore',
        department: 'Research & Analytics'
      },
      {
        id: 'user-4',
        name: 'Emily Davis',
        email: 'emily.davis@company.com',
        role: 'viewer',
        status: 'suspended',
        lastLogin: '2024-01-18T09:30:00Z',
        createdAt: '2024-01-15T00:00:00Z',
        totalTrades: 0,
        portfolioValue: 0,
        riskLevel: 'low',
        permissions: ['READ_ONLY_ACCESS'],
        location: 'Toronto, CA',
        department: 'Compliance'
      }
    ];

    const sampleSystemHealth: SystemHealth[] = [
      {
        service: 'Trading Engine',
        status: 'healthy',
        uptime: '99.98%',
        cpu: 45.2,
        memory: 68.7,
        responseTime: 12,
        lastCheck: '2024-01-20T16:00:00Z',
        version: 'v2.4.1'
      },
      {
        service: 'Market Data Feed',
        status: 'healthy',
        uptime: '99.95%',
        cpu: 32.1,
        memory: 54.3,
        responseTime: 8,
        lastCheck: '2024-01-20T16:00:00Z',
        version: 'v1.8.3'
      },
      {
        service: 'Risk Manager',
        status: 'warning',
        uptime: '99.12%',
        cpu: 78.9,
        memory: 82.1,
        responseTime: 45,
        lastCheck: '2024-01-20T16:00:00Z',
        version: 'v3.1.0'
      },
      {
        service: 'Database Cluster',
        status: 'healthy',
        uptime: '99.99%',
        cpu: 23.4,
        memory: 45.8,
        responseTime: 15,
        lastCheck: '2024-01-20T16:00:00Z',
        version: 'PostgreSQL 15.2'
      },
      {
        service: 'API Gateway',
        status: 'healthy',
        uptime: '99.87%',
        cpu: 38.7,
        memory: 62.3,
        responseTime: 22,
        lastCheck: '2024-01-20T16:00:00Z',
        version: 'v1.2.8'
      },
      {
        service: 'Analytics Engine',
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
        user: 'john.anderson@company.com',
        action: 'USER_SUSPENDED',
        resource: 'users/emily.davis',
        details: 'User suspended due to policy violation',
        ipAddress: '192.168.1.145',
        result: 'success',
        severity: 'high'
      },
      {
        id: 'log-2',
        timestamp: '2024-01-20T15:30:12Z',
        user: 'sarah.chen@company.com',
        action: 'LARGE_ORDER_EXECUTED',
        resource: 'orders/ord-789456',
        details: 'Executed order for 50,000 shares of AAPL',
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
        details: 'Updated portfolio VaR limit from $45K to $50K',
        ipAddress: '127.0.0.1',
        result: 'success',
        severity: 'medium'
      },
      {
        id: 'log-4',
        timestamp: '2024-01-20T14:58:33Z',
        user: 'michael.rodriguez@company.com',
        action: 'LOGIN_FAILED',
        resource: 'auth/login',
        details: 'Failed login attempt - invalid credentials',
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
        description: 'Maximum order size in USD',
        modified: '2024-01-20T10:30:00Z',
        modifiedBy: 'john.anderson@company.com',
        requiresRestart: false
      },
      {
        id: 'config-2',
        category: 'Risk',
        key: 'portfolio_var_limit',
        value: '50000',
        type: 'number',
        description: 'Portfolio Value at Risk limit in USD',
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
        description: 'User session timeout in seconds',
        modified: '2024-01-19T14:20:00Z',
        modifiedBy: 'john.anderson@company.com',
        requiresRestart: true
      },
      {
        id: 'config-4',
        category: 'Security',
        key: 'enable_2fa',
        value: 'true',
        type: 'boolean',
        description: 'Require two-factor authentication',
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
    return new Date(timestamp).toLocaleString();
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
            Admin Panel
          </h1>
          <p className="text-muted-foreground">
            System administration, user management, and platform configuration
          </p>
        </div>
        
        {/* Quick Stats */}
        <div className="flex gap-4">
          <Card className="w-32">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-blue-600">{totalUsers}</div>
              <div className="text-xs text-gray-500">Total Users</div>
            </CardContent>
          </Card>
          <Card className="w-32">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-green-600">{activeUsers}</div>
              <div className="text-xs text-gray-500">Active</div>
            </CardContent>
          </Card>
          <Card className="w-32">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-green-600">{healthyServices}</div>
              <div className="text-xs text-gray-500">Healthy Services</div>
            </CardContent>
          </Card>
          <Card className="w-32">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-red-600">{criticalIssues}</div>
              <div className="text-xs text-gray-500">Critical Issues</div>
            </CardContent>
          </Card>
        </div>
      </div>

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-7">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="users">Users</TabsTrigger>
          <TabsTrigger value="system">System Health</TabsTrigger>
          <TabsTrigger value="audit">Audit Logs</TabsTrigger>
          <TabsTrigger value="config">Configuration</TabsTrigger>
          <TabsTrigger value="tools">Tools</TabsTrigger>
          <TabsTrigger value="startup">استارتاپ‌تراکر</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* System Status Overview */}
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Monitor className="w-5 h-5" />
                  System Status
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
                        {service.status}
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
                  User Activity
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Active Users</span>
                    <span className="font-medium">{activeUsers} / {totalUsers}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Suspended</span>
                    <span className="font-medium">{users.filter(u => u.status === 'suspended').length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Pending</span>
                    <span className="font-medium">{users.filter(u => u.status === 'pending').length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Total Trades Today</span>
                    <span className="font-medium">2,337</span>
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
                Recent Administrative Activity
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
                      {log.result}
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
                  User Management
                </div>
                <Button className="flex items-center gap-2">
                  <Plus className="w-4 h-4" />
                  Add User
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-4">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                    <Input
                      placeholder="Search users..."
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
                  <option value="all">All Roles</option>
                  <option value="admin">Admin</option>
                  <option value="trader">Trader</option>
                  <option value="analyst">Analyst</option>
                  <option value="viewer">Viewer</option>
                </select>
                <select 
                  value={selectedStatus} 
                  onChange={(e) => setSelectedStatus(e.target.value)}
                  className="px-3 py-2 border rounded-md"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="inactive">Inactive</option>
                  <option value="suspended">Suspended</option>
                  <option value="pending">Pending</option>
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
                          {user.role}
                        </Badge>
                        <Badge className={getStatusColor(user.status)}>
                          {user.status}
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
                        <div className="text-gray-500">Department</div>
                        <div className="font-medium">{user.department}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Location</div>
                        <div className="font-medium">{user.location}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Total Trades</div>
                        <div className="font-medium">{user.totalTrades.toLocaleString()}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Portfolio Value</div>
                        <div className="font-medium">{formatCurrency(user.portfolioValue)}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Last Login</div>
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
                  System Health Monitor
                </div>
                <Button variant="outline" size="sm">
                  <RefreshCw className="w-4 h-4 mr-1" />
                  Refresh
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
                          {service.status}
                        </Badge>
                        <Badge variant="outline">{service.version}</Badge>
                      </div>
                      <div className="text-sm text-gray-500">
                        Last check: {formatTimestamp(service.lastCheck)}
                      </div>
                    </div>
                    
                    <div className="grid gap-4 md:grid-cols-5 text-sm">
                      <div>
                        <div className="text-gray-500">Uptime</div>
                        <div className="font-medium">{service.uptime}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">CPU Usage</div>
                        <div className="font-medium">{service.cpu}%</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Memory</div>
                        <div className="font-medium">{service.memory}%</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Response Time</div>
                        <div className="font-medium">{service.responseTime}ms</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Actions</div>
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
                System Configuration
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
                            Requires Restart
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
                        Modified: {formatTimestamp(setting.modified)} by {setting.modifiedBy}
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
                  Database Tools
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full justify-start">
                  <Download className="w-4 h-4 mr-2" />
                  Export Database Backup
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Upload className="w-4 h-4 mr-2" />
                  Import Data
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Wrench className="w-4 h-4 mr-2" />
                  Database Maintenance
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Performance Analytics
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  Security Tools
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full justify-start">
                  <Key className="w-4 h-4 mr-2" />
                  Rotate API Keys
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Lock className="w-4 h-4 mr-2" />
                  Force Password Reset
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <AlertTriangle className="w-4 h-4 mr-2" />
                  Security Scan
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <FileText className="w-4 h-4 mr-2" />
                  Security Report
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Monitor className="w-5 h-5" />
                  System Tools
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full justify-start">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Restart Services
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Activity className="w-4 h-4 mr-2" />
                  Health Check
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Archive className="w-4 h-4 mr-2" />
                  Archive Old Logs
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Target className="w-4 h-4 mr-2" />
                  Performance Test
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bell className="w-5 h-5" />
                  Notification Tools
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full justify-start">
                  <Mail className="w-4 h-4 mr-2" />
                  Send System Notice
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <AlertCircle className="w-4 h-4 mr-2" />
                  Emergency Alert
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Bell className="w-4 h-4 mr-2" />
                  Test Notifications
                </Button>
                <Button className="w-full justify-start" variant="outline">
                  <Settings className="w-4 h-4 mr-2" />
                  Alert Settings
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