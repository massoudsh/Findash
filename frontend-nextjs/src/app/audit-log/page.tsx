'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  FileText,
  Search,
  Filter,
  Download,
  Calendar,
  Clock,
  User,
  Shield,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Settings,
  Database,
  Activity,
  Globe,
  Smartphone,
  Monitor,
  MapPin,
  RefreshCw,
  MoreHorizontal,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Archive,
  AlertCircle,
  Info,
  Zap,
  Target,
  Users,
  Building,
  CreditCard,
  LogOut,
  LogIn,
  Edit,
  Trash2,
  Plus,
  Minus,
  ArrowRight,
  ArrowLeft,
  BarChart3
} from 'lucide-react';

interface AuditEvent {
  id: string;
  timestamp: string;
  userId: string;
  username: string;
  userRole: string;
  action: string;
  category: 'Authentication' | 'Trading' | 'Account' | 'System' | 'Compliance' | 'Security' | 'API';
  resource: string;
  details: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  result: 'success' | 'failure' | 'warning';
  ipAddress: string;
  userAgent: string;
  location: string;
  sessionId: string;
  metadata: Record<string, any>;
  riskScore: number;
  complianceFlags: string[];
}

interface AuditSummary {
  totalEvents: number;
  todayEvents: number;
  failedActions: number;
  highRiskEvents: number;
  uniqueUsers: number;
  topActions: { action: string; count: number }[];
  severityBreakdown: { severity: string; count: number }[];
}

const CATEGORY_LABELS: Record<string, string> = {
  Authentication: 'احراز هویت',
  Trading: 'معاملات',
  Account: 'حساب کاربری',
  System: 'سیستم',
  Compliance: 'انطباق',
  Security: 'امنیت',
  API: 'API',
};

const SEVERITY_LABELS: Record<string, string> = {
  low: 'کم',
  medium: 'متوسط',
  high: 'زیاد',
  critical: 'بحرانی',
};

const RESULT_LABELS: Record<string, string> = {
  success: 'موفق',
  failure: 'ناموفق',
  warning: 'هشدار',
};

export default function AuditLogPage() {
  const [selectedTab, setSelectedTab] = useState('events');
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [filteredEvents, setFilteredEvents] = useState<AuditEvent[]>([]);
  const [summary, setSummary] = useState<AuditSummary | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedSeverity, setSelectedSeverity] = useState('all');
  const [selectedResult, setSelectedResult] = useState('all');
  const [dateRange, setDateRange] = useState('today');
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const eventsPerPage = 20;

  useEffect(() => {
    // Generate sample audit events
    const sampleEvents: AuditEvent[] = [
      {
        id: 'audit-001',
        timestamp: '2024-01-20T15:45:23Z',
        userId: 'user-123',
        username: 'john.anderson@company.com',
        userRole: 'trader',
        action: 'ORDER_PLACED',
        category: 'Trading',
        resource: 'orders/ord-789456',
        details: 'ثبت سفارش بازار برای ۱۰۰۰ سهم AAPL با قیمت ۱۸۵.۵۰ دلار',
        severity: 'medium',
        result: 'success',
        ipAddress: '192.168.1.145',
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        location: 'نیویورک، آمریکا',
        sessionId: 'sess-abc123',
        metadata: { symbol: 'AAPL', quantity: 1000, price: 185.50, orderType: 'market' },
        riskScore: 65,
        complianceFlags: ['LARGE_ORDER']
      },
      {
        id: 'audit-002',
        timestamp: '2024-01-20T15:42:15Z',
        userId: 'user-456',
        username: 'sarah.chen@company.com',
        userRole: 'admin',
        action: 'USER_ROLE_CHANGED',
        category: 'Security',
        resource: 'users/user-789',
        details: 'نقش کاربر emily.davis@company.com از viewer به trader تغییر یافت',
        severity: 'high',
        result: 'success',
        ipAddress: '10.0.0.234',
        userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        location: 'لندن، انگلستان',
        sessionId: 'sess-def456',
        metadata: { targetUser: 'emily.davis@company.com', oldRole: 'viewer', newRole: 'trader' },
        riskScore: 85,
        complianceFlags: ['PRIVILEGE_ESCALATION', 'ADMIN_ACTION']
      },
      {
        id: 'audit-003',
        timestamp: '2024-01-20T15:38:42Z',
        userId: 'user-789',
        username: 'michael.rodriguez@company.com',
        userRole: 'analyst',
        action: 'LOGIN_FAILED',
        category: 'Authentication',
        resource: 'auth/login',
        details: 'تلاش ورود ناموفق - رمز عبور نامعتبر',
        severity: 'medium',
        result: 'failure',
        ipAddress: '203.45.67.89',
        userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15',
        location: 'سنگاپور',
        sessionId: '',
        metadata: { attemptNumber: 3, lockoutTriggered: false },
        riskScore: 45,
        complianceFlags: ['FAILED_AUTH']
      },
      {
        id: 'audit-004',
        timestamp: '2024-01-20T15:35:18Z',
        userId: 'system',
        username: 'system',
        userRole: 'system',
        action: 'RISK_LIMIT_EXCEEDED',
        category: 'Compliance',
        resource: 'risk/portfolio-var',
        details: 'محدودیت VaR پرتفوی نقض شد: ۵۲,۰۰۰ دلار (محدودیت: ۵۰,۰۰۰ دلار)',
        severity: 'critical',
        result: 'warning',
        ipAddress: '127.0.0.1',
        userAgent: 'System/Internal',
        location: 'سیستم',
        sessionId: 'sys-001',
        metadata: { currentVaR: 52000, limit: 50000, portfolioId: 'port-123' },
        riskScore: 95,
        complianceFlags: ['RISK_BREACH', 'AUTOMATIC_ALERT']
      },
      {
        id: 'audit-005',
        timestamp: '2024-01-20T15:30:45Z',
        userId: 'user-123',
        username: 'john.anderson@company.com',
        userRole: 'trader',
        action: 'POSITION_CLOSED',
        category: 'Trading',
        resource: 'positions/pos-456789',
        details: 'پوزیشن بسته شد: ۵۰۰ سهم TSLA، سود و زیان: +۱۲,۵۰۰ دلار',
        severity: 'low',
        result: 'success',
        ipAddress: '192.168.1.145',
        userAgent: 'TradingApp/1.0 (Mobile)',
        location: 'نیویورک، آمریکا',
        sessionId: 'sess-abc123',
        metadata: { symbol: 'TSLA', quantity: 500, pnl: 12500, reason: 'manual' },
        riskScore: 25,
        complianceFlags: []
      },
      {
        id: 'audit-006',
        timestamp: '2024-01-20T15:25:33Z',
        userId: 'user-999',
        username: 'suspicious.user@external.com',
        userRole: 'viewer',
        action: 'API_RATE_LIMIT_EXCEEDED',
        category: 'API',
        resource: 'api/market-data',
        details: 'محدودیت نرخ API نقض شد: ۱۰۰۰ درخواست در ۱ دقیقه (محدودیت: ۵۰۰)',
        severity: 'high',
        result: 'failure',
        ipAddress: '45.67.89.123',
        userAgent: 'Python/requests',
        location: 'نامشخص',
        sessionId: 'api-xyz789',
        metadata: { requestCount: 1000, timeWindow: '1m', endpoint: '/api/market-data' },
        riskScore: 78,
        complianceFlags: ['RATE_LIMIT_ABUSE', 'SUSPICIOUS_ACTIVITY']
      },
      {
        id: 'audit-007',
        timestamp: '2024-01-20T15:20:12Z',
        userId: 'user-456',
        username: 'sarah.chen@company.com',
        userRole: 'admin',
        action: 'CONFIG_UPDATED',
        category: 'System',
        resource: 'config/trading-limits',
        details: 'حداکثر حجم سفارش از ۵۰۰ هزار به ۱ میلیون دلار افزایش یافت',
        severity: 'medium',
        result: 'success',
        ipAddress: '10.0.0.234',
        userAgent: 'Chrome/120.0.0.0',
        location: 'لندن، انگلستان',
        sessionId: 'sess-def456',
        metadata: { setting: 'max_order_size', oldValue: 500000, newValue: 1000000 },
        riskScore: 55,
        complianceFlags: ['CONFIG_CHANGE', 'ADMIN_ACTION']
      },
      {
        id: 'audit-008',
        timestamp: '2024-01-20T15:15:27Z',
        userId: 'user-321',
        username: 'compliance.officer@company.com',
        userRole: 'compliance',
        action: 'REPORT_GENERATED',
        category: 'Compliance',
        resource: 'reports/daily-trading',
        details: 'گزارش انطباق معاملات روزانه برای تاریخ ۲۰۲۴-۰۱-۲۰ تولید شد',
        severity: 'low',
        result: 'success',
        ipAddress: '172.16.0.100',
        userAgent: 'Safari/17.2',
        location: 'تورنتو، کانادا',
        sessionId: 'sess-comp001',
        metadata: { reportType: 'daily-trading', period: '2024-01-20', recordCount: 2847 },
        riskScore: 10,
        complianceFlags: ['COMPLIANCE_REPORT']
      }
    ];

    setEvents(sampleEvents);
    setFilteredEvents(sampleEvents);

    // Generate summary
    const summaryData: AuditSummary = {
      totalEvents: sampleEvents.length,
      todayEvents: sampleEvents.filter(e => e.timestamp.startsWith('2024-01-20')).length,
      failedActions: sampleEvents.filter(e => e.result === 'failure').length,
      highRiskEvents: sampleEvents.filter(e => e.riskScore >= 70).length,
      uniqueUsers: new Set(sampleEvents.map(e => e.userId)).size,
      topActions: [
        { action: 'ORDER_PLACED', count: 245 },
        { action: 'LOGIN_SUCCESS', count: 189 },
        { action: 'POSITION_CLOSED', count: 156 },
        { action: 'LOGIN_FAILED', count: 43 },
        { action: 'CONFIG_UPDATED', count: 12 }
      ],
      severityBreakdown: [
        { severity: 'low', count: 45 },
        { severity: 'medium', count: 32 },
        { severity: 'high', count: 18 },
        { severity: 'critical', count: 5 }
      ]
    };
    setSummary(summaryData);
  }, []);

  useEffect(() => {
    let filtered = events;

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(event =>
        event.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
        event.action.toLowerCase().includes(searchTerm.toLowerCase()) ||
        event.details.toLowerCase().includes(searchTerm.toLowerCase()) ||
        event.resource.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply category filter
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(event => event.category === selectedCategory);
    }

    // Apply severity filter
    if (selectedSeverity !== 'all') {
      filtered = filtered.filter(event => event.severity === selectedSeverity);
    }

    // Apply result filter
    if (selectedResult !== 'all') {
      filtered = filtered.filter(event => event.result === selectedResult);
    }

    setFilteredEvents(filtered);
    setCurrentPage(1);
  }, [events, searchTerm, selectedCategory, selectedSeverity, selectedResult]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'high': return 'bg-orange-100 text-orange-800';
      case 'critical': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getResultColor = (result: string) => {
    switch (result) {
      case 'success': return 'bg-green-100 text-green-800';
      case 'failure': return 'bg-red-100 text-red-800';
      case 'warning': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Authentication': return <LogIn className="w-4 h-4" />;
      case 'Trading': return <DollarSign className="w-4 h-4" />;
      case 'Account': return <User className="w-4 h-4" />;
      case 'System': return <Settings className="w-4 h-4" />;
      case 'Compliance': return <Shield className="w-4 h-4" />;
      case 'Security': return <Lock className="w-4 h-4" />;
      case 'API': return <Globe className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('fa-IR');
  };

  const getRiskLevel = (score: number) => {
    if (score >= 80) return { level: 'بحرانی', color: 'text-red-600' };
    if (score >= 60) return { level: 'زیاد', color: 'text-orange-600' };
    if (score >= 30) return { level: 'متوسط', color: 'text-yellow-600' };
    return { level: 'کم', color: 'text-green-600' };
  };

  const exportToCSV = () => {
    const headers = ['زمان', 'کاربر', 'عملیات', 'دسته', 'نتیجه', 'جزئیات', 'آدرس IP', 'امتیاز ریسک'];
    const csvContent = [
      headers.join(','),
      ...filteredEvents.map(event => [
        event.timestamp,
        event.username,
        event.action,
        event.category,
        event.result,
        `"${event.details}"`,
        event.ipAddress,
        event.riskScore
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audit-log-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  // Pagination
  const totalPages = Math.ceil(filteredEvents.length / eventsPerPage);
  const startIndex = (currentPage - 1) * eventsPerPage;
  const paginatedEvents = filteredEvents.slice(startIndex, startIndex + eventsPerPage);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <FileText className="w-8 h-8 text-blue-600" />
            لاگ ممیزی
          </h1>
          <p className="text-muted-foreground">
            ردیابی جامع فعالیت‌های کاربران و رویدادهای سیستم برای انطباق و امنیت
          </p>
        </div>

        <div className="flex gap-3">
          <Button variant="outline" onClick={exportToCSV} className="flex items-center gap-2">
            <Download className="w-4 h-4" />
            خروجی CSV
          </Button>
          <Button className="flex items-center gap-2">
            <RefreshCw className="w-4 h-4" />
            بازخوانی
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-6">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">کل رویدادها</p>
                  <p className="text-2xl font-bold">{summary.totalEvents.toLocaleString()}</p>
                </div>
                <Activity className="w-8 h-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">امروز</p>
                  <p className="text-2xl font-bold">{summary.todayEvents}</p>
                </div>
                <Calendar className="w-8 h-8 text-green-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">اقدامات ناموفق</p>
                  <p className="text-2xl font-bold">{summary.failedActions}</p>
                </div>
                <XCircle className="w-8 h-8 text-red-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">ریسک بالا</p>
                  <p className="text-2xl font-bold">{summary.highRiskEvents}</p>
                </div>
                <AlertTriangle className="w-8 h-8 text-orange-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">کاربران یکتا</p>
                  <p className="text-2xl font-bold">{summary.uniqueUsers}</p>
                </div>
                <Users className="w-8 h-8 text-purple-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">انطباق</p>
                  <p className="text-2xl font-bold text-green-600">✓</p>
                </div>
                <Shield className="w-8 h-8 text-green-600" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="events">رویدادهای ممیزی</TabsTrigger>
          <TabsTrigger value="analytics">تحلیل</TabsTrigger>
          <TabsTrigger value="compliance">انطباق</TabsTrigger>
          <TabsTrigger value="alerts">هشدارهای امنیتی</TabsTrigger>
        </TabsList>

        <TabsContent value="events" className="space-y-4">
          {/* Filters */}
          <Card>
            <CardContent className="p-6">
              <div className="grid gap-4 md:grid-cols-6">
                <div className="md:col-span-2">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                    <Input
                      placeholder="جستجوی رویدادها..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>

                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="px-3 py-2 border rounded-md"
                >
                  <option value="all">همه دسته‌ها</option>
                  <option value="Authentication">احراز هویت</option>
                  <option value="Trading">معاملات</option>
                  <option value="Account">حساب کاربری</option>
                  <option value="System">سیستم</option>
                  <option value="Compliance">انطباق</option>
                  <option value="Security">امنیت</option>
                  <option value="API">API</option>
                </select>

                <select
                  value={selectedSeverity}
                  onChange={(e) => setSelectedSeverity(e.target.value)}
                  className="px-3 py-2 border rounded-md"
                >
                  <option value="all">همه سطوح شدت</option>
                  <option value="low">کم</option>
                  <option value="medium">متوسط</option>
                  <option value="high">زیاد</option>
                  <option value="critical">بحرانی</option>
                </select>

                <select
                  value={selectedResult}
                  onChange={(e) => setSelectedResult(e.target.value)}
                  className="px-3 py-2 border rounded-md"
                >
                  <option value="all">همه نتایج</option>
                  <option value="success">موفق</option>
                  <option value="failure">ناموفق</option>
                  <option value="warning">هشدار</option>
                </select>

                <select
                  value={dateRange}
                  onChange={(e) => setDateRange(e.target.value)}
                  className="px-3 py-2 border rounded-md"
                >
                  <option value="today">امروز</option>
                  <option value="week">این هفته</option>
                  <option value="month">این ماه</option>
                  <option value="all">همه زمان‌ها</option>
                </select>
              </div>

              <div className="mt-4 flex items-center justify-between">
                <div className="text-sm text-gray-600">
                  نمایش {startIndex + 1}-{Math.min(startIndex + eventsPerPage, filteredEvents.length)} از {filteredEvents.length} رویداد
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    disabled={currentPage === 1}
                  >
                    <ArrowLeft className="w-4 h-4" />
                  </Button>
                  <span className="px-3 py-1 text-sm">
                    {currentPage} از {totalPages}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                    disabled={currentPage === totalPages}
                  >
                    <ArrowRight className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Events List */}
          <Card>
            <CardContent className="p-0">
              <div className="space-y-1">
                {paginatedEvents.map((event) => (
                  <div key={event.id} className="border-b last:border-b-0">
                    <div
                      className="p-4 hover:bg-gray-50 cursor-pointer"
                      onClick={() => setExpandedEvent(expandedEvent === event.id ? null : event.id)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4 flex-1">
                          <div className="flex items-center gap-2">
                            {getCategoryIcon(event.category)}
                            <span className="text-xs text-gray-500">{CATEGORY_LABELS[event.category] ?? event.category}</span>
                          </div>

                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-medium truncate">{event.action.replace('_', ' ')}</span>
                              <Badge className={getSeverityColor(event.severity)}>
                                {SEVERITY_LABELS[event.severity] ?? event.severity}
                              </Badge>
                              <Badge className={getResultColor(event.result)}>
                                {RESULT_LABELS[event.result] ?? event.result}
                              </Badge>
                            </div>
                            <div className="text-sm text-gray-600 truncate">{event.details}</div>
                            <div className="flex items-center gap-4 text-xs text-gray-500 mt-1">
                              <span>{event.username}</span>
                              <span>{formatTimestamp(event.timestamp)}</span>
                              <span>{event.ipAddress}</span>
                              <span className={getRiskLevel(event.riskScore).color}>
                                ریسک: {getRiskLevel(event.riskScore).level}
                              </span>
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          {event.complianceFlags.length > 0 && (
                            <Badge variant="outline" className="text-orange-600">
                              {event.complianceFlags.length} پرچم
                            </Badge>
                          )}
                          {expandedEvent === event.id ?
                            <ChevronDown className="w-4 h-4" /> :
                            <ChevronRight className="w-4 h-4" />
                          }
                        </div>
                      </div>
                    </div>

                    {expandedEvent === event.id && (
                      <div className="px-4 pb-4 bg-gray-50 border-t">
                        <div className="grid gap-4 md:grid-cols-2 mt-4">
                          <div className="space-y-2">
                            <h4 className="font-medium text-sm">جزئیات رویداد</h4>
                            <div className="space-y-1 text-sm">
                              <div className="flex justify-between">
                                <span className="text-gray-600">شناسه رویداد:</span>
                                <span className="font-mono">{event.id}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600">منبع:</span>
                                <span className="font-mono">{event.resource}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600">شناسه جلسه:</span>
                                <span className="font-mono">{event.sessionId}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600">امتیاز ریسک:</span>
                                <span className={getRiskLevel(event.riskScore).color}>
                                  {event.riskScore}/100
                                </span>
                              </div>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <h4 className="font-medium text-sm">کاربر و موقعیت</h4>
                            <div className="space-y-1 text-sm">
                              <div className="flex justify-between">
                                <span className="text-gray-600">کاربر:</span>
                                <span>{event.username}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600">نقش:</span>
                                <span className="capitalize">{event.userRole}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600">موقعیت:</span>
                                <span>{event.location}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600">مرورگر/دستگاه:</span>
                                <span className="truncate">{event.userAgent.split(' ')[0]}</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        {event.complianceFlags.length > 0 && (
                          <div className="mt-4">
                            <h4 className="font-medium text-sm mb-2">پرچم‌های انطباق</h4>
                            <div className="flex gap-2 flex-wrap">
                              {event.complianceFlags.map(flag => (
                                <Badge key={flag} variant="outline" className="text-orange-600">
                                  {flag}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}

                        {Object.keys(event.metadata).length > 0 && (
                          <div className="mt-4">
                            <h4 className="font-medium text-sm mb-2">فراداده</h4>
                            <div className="bg-gray-100 p-3 rounded text-xs font-mono">
                              {JSON.stringify(event.metadata, null, 2)}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid gap-6 md:grid-cols-2">
            {summary && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="w-5 h-5" />
                      پراستفاده‌ترین عملیات‌ها
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {summary.topActions.map((action, index) => (
                        <div key={action.action} className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-blue-600">#{index + 1}</span>
                            <span className="text-sm">{action.action.replace('_', ' ')}</span>
                          </div>
                          <span className="text-sm font-semibold">{action.count}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="w-5 h-5" />
                      تفکیک بر اساس شدت
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {summary.severityBreakdown.map((item) => (
                        <div key={item.severity} className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Badge className={getSeverityColor(item.severity)}>
                              {SEVERITY_LABELS[item.severity] ?? item.severity}
                            </Badge>
                          </div>
                          <span className="text-sm font-semibold">{item.count}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </TabsContent>

        <TabsContent value="compliance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="w-5 h-5" />
                داشبورد انطباق
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-4 md:grid-cols-3">
                <div className="text-center p-4 border rounded-lg">
                  <div className="text-2xl font-bold text-green-600">۹۸.۵٪</div>
                  <div className="text-sm text-gray-600">امتیاز انطباق</div>
                </div>
                <div className="text-center p-4 border rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">۲۴۷</div>
                  <div className="text-sm text-gray-600">رویدادهای پرچم‌گذاری‌شده</div>
                </div>
                <div className="text-center p-4 border rounded-lg">
                  <div className="text-2xl font-bold text-orange-600">۱۲</div>
                  <div className="text-sm text-gray-600">موارد باز</div>
                </div>
              </div>

              <div className="space-y-3">
                <h3 className="font-medium">پرچم‌های انطباق اخیر</h3>
                {events.filter(e => e.complianceFlags.length > 0).slice(0, 5).map(event => (
                  <div key={event.id} className="p-3 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{event.action.replace('_', ' ')}</span>
                      <Badge className={getSeverityColor(event.severity)}>
                        {SEVERITY_LABELS[event.severity] ?? event.severity}
                      </Badge>
                    </div>
                    <div className="text-sm text-gray-600 mb-2">{event.details}</div>
                    <div className="flex gap-2 flex-wrap">
                      {event.complianceFlags.map(flag => (
                        <Badge key={flag} variant="outline" className="text-orange-600">
                          {flag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                هشدارهای امنیتی
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 border border-red-200 bg-red-50 rounded-lg">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                    <div className="flex-1">
                      <div className="font-medium text-red-800">رویداد ریسک بحرانی شناسایی شد</div>
                      <div className="text-sm text-red-700 mt-1">
                        محدودیت VaR پرتفوی به میزان ۴٪ نقض شد. بستن خودکار پوزیشن آغاز شد.
                      </div>
                      <div className="text-xs text-red-600 mt-2">
                        ۱۵:۳۵ UTC • امتیاز ریسک: ۹۵/۱۰۰
                      </div>
                    </div>
                  </div>
                </div>

                <div className="p-4 border border-orange-200 bg-orange-50 rounded-lg">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-orange-600 mt-0.5" />
                    <div className="flex-1">
                      <div className="font-medium text-orange-800">فعالیت مشکوک API</div>
                      <div className="text-sm text-orange-700 mt-1">
                        محدودیت نرخ API از یک IP خارجی نقض شد. احتمال سوءاستفاده شناسایی شد.
                      </div>
                      <div className="text-xs text-orange-600 mt-2">
                        ۱۵:۲۵ UTC • امتیاز ریسک: ۷۸/۱۰۰
                      </div>
                    </div>
                  </div>
                </div>

                <div className="p-4 border border-yellow-200 bg-yellow-50 rounded-lg">
                  <div className="flex items-start gap-3">
                    <Info className="w-5 h-5 text-yellow-600 mt-0.5" />
                    <div className="flex-1">
                      <div className="font-medium text-yellow-800">تشدید سطح دسترسی</div>
                      <div className="text-sm text-yellow-700 mt-1">
                        کاربر ادمین نقش trader را به یک حساب viewer اعطا کرد. بررسی توصیه می‌شود.
                      </div>
                      <div className="text-xs text-yellow-600 mt-2">
                        ۱۵:۴۲ UTC • امتیاز ریسک: ۸۵/۱۰۰
                      </div>
                    </div>
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