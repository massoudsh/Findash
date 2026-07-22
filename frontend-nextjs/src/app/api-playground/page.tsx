'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Code2,
  Play,
  Copy,
  Download,
  Settings,
  Key,
  Globe,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  FileText,
  Terminal,
  Database,
  Zap,
  Shield,
  Eye,
  EyeOff,
  Plus,
  Minus,
  RotateCcw,
  Save,
  History,
  BookOpen,
  TrendingUp,
  DollarSign,
  BarChart3,
  Activity
} from 'lucide-react';

interface APIEndpoint {
  id: string;
  name: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  path: string;
  description: string;
  category: string;
  requiresAuth: boolean;
  parameters: Parameter[];
  headers: Header[];
  requestBody?: string;
  responseExample: string;
}

interface Parameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'array';
  required: boolean;
  description: string;
  example: string;
}

interface Header {
  name: string;
  value: string;
  required: boolean;
}

interface APIRequest {
  id: string;
  endpoint: APIEndpoint;
  timestamp: string;
  status: number;
  responseTime: number;
  response: string;
  success: boolean;
}

const CATEGORY_LABELS: Record<string, string> = {
  'Market Data': 'داده بازار',
  'Portfolio': 'پرتفوی',
  'Trading': 'معاملات',
  'Risk Management': 'مدیریت ریسک',
  'Analytics': 'تحلیل',
};

export default function APIPlaygroundPage() {
  const [selectedEndpoint, setSelectedEndpoint] = useState<APIEndpoint | null>(null);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [requestHistory, setRequestHistory] = useState<APIRequest[]>([]);
  const [loading, setLoading] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [baseUrl, setBaseUrl] = useState('http://localhost:8000');
  const [customHeaders, setCustomHeaders] = useState<Header[]>([]);
  const [requestBody, setRequestBody] = useState('');
  const [response, setResponse] = useState('');
  const [responseStatus, setResponseStatus] = useState<number | null>(null);
  const [responseTime, setResponseTime] = useState<number | null>(null);

  // API Endpoints Configuration
  const apiEndpoints: APIEndpoint[] = [
    // Market Data Endpoints
    {
      id: 'market-data-current',
      name: 'دریافت داده لحظه‌ای بازار',
      method: 'GET',
      path: '/api/market-data/current',
      description: 'دریافت داده لحظه‌ای بازار برای نمادهای مشخص‌شده',
      category: 'Market Data',
      requiresAuth: true,
      parameters: [
        { name: 'symbols', type: 'string', required: true, description: 'فهرست نمادها با کاما جدا شده', example: 'AAPL,TSLA,MSFT' },
        { name: 'fields', type: 'string', required: false, description: 'فیلدهای خاص برای دریافت', example: 'price,volume,change' }
      ],
      headers: [
        { name: 'Authorization', value: 'Bearer {token}', required: true },
        { name: 'Content-Type', value: 'application/json', required: true }
      ],
      responseExample: `{
  "data": {
    "AAPL": {
      "symbol": "AAPL",
      "price": 177.45,
      "change": -0.78,
      "changePercent": -0.44,
      "volume": 45234567,
      "high": 179.20,
      "low": 176.30,
      "open": 178.50,
      "marketCap": 2750000000000,
      "timestamp": "2024-01-20T16:00:00Z"
    }
  },
  "status": "success",
  "timestamp": "2024-01-20T16:00:01Z"
}`
    },
    {
      id: 'market-data-historical',
      name: 'دریافت داده تاریخی',
      method: 'GET',
      path: '/api/market-data/historical',
      description: 'دریافت داده تاریخی بازار با بازه‌های زمانی مختلف',
      category: 'Market Data',
      requiresAuth: true,
      parameters: [
        { name: 'symbol', type: 'string', required: true, description: 'نماد سهم', example: 'AAPL' },
        { name: 'period', type: 'string', required: true, description: 'بازه زمانی', example: '1d,1w,1m,3m,1y' },
        { name: 'interval', type: 'string', required: false, description: 'فاصله داده‌ها', example: '1m,5m,1h,1d' }
      ],
      headers: [
        { name: 'Authorization', value: 'Bearer {token}', required: true }
      ],
      responseExample: `{
  "data": {
    "symbol": "AAPL",
    "period": "1w",
    "interval": "1d",
    "prices": [
      {
        "timestamp": "2024-01-15T00:00:00Z",
        "open": 175.20,
        "high": 178.45,
        "low": 174.80,
        "close": 177.45,
        "volume": 52345678
      }
    ]
  }
}`
    },
    // Portfolio Endpoints
    {
      id: 'portfolio-list',
      name: 'فهرست پرتفوی‌ها',
      method: 'GET',
      path: '/api/portfolio',
      description: 'دریافت همه پرتفوی‌های کاربر احراز هویت‌شده',
      category: 'Portfolio',
      requiresAuth: true,
      parameters: [
        { name: 'include_positions', type: 'boolean', required: false, description: 'شامل جزئیات پوزیشن‌ها', example: 'true' }
      ],
      headers: [
        { name: 'Authorization', value: 'Bearer {token}', required: true }
      ],
      responseExample: `{
  "portfolios": [
    {
      "id": "port_123",
      "name": "Growth Portfolio",
      "totalValue": 50000.00,
      "totalPnL": 2450.30,
      "totalPnLPercent": 5.14,
      "cash": 5000.00,
      "positions": [
        {
          "symbol": "AAPL",
          "quantity": 50,
          "averagePrice": 175.00,
          "currentPrice": 177.45,
          "marketValue": 8872.50,
          "unrealizedPnL": 122.50
        }
      ]
    }
  ]
}`
    },
    {
      id: 'portfolio-create',
      name: 'ایجاد پرتفوی',
      method: 'POST',
      path: '/api/portfolio',
      description: 'ایجاد یک پرتفوی جدید',
      category: 'Portfolio',
      requiresAuth: true,
      parameters: [],
      headers: [
        { name: 'Authorization', value: 'Bearer {token}', required: true },
        { name: 'Content-Type', value: 'application/json', required: true }
      ],
      requestBody: `{
  "name": "New Portfolio",
  "description": "My investment portfolio",
  "initialCash": 10000.00,
  "riskLevel": "moderate"
}`,
      responseExample: `{
  "portfolio": {
    "id": "port_456",
    "name": "New Portfolio",
    "description": "My investment portfolio",
    "totalValue": 10000.00,
    "cash": 10000.00,
    "created": "2024-01-20T16:00:00Z"
  }
}`
    },
    // Trading Endpoints
    {
      id: 'orders-list',
      name: 'فهرست سفارش‌ها',
      method: 'GET',
      path: '/api/trading/orders',
      description: 'دریافت همه سفارش‌ها با امکان فیلتر',
      category: 'Trading',
      requiresAuth: true,
      parameters: [
        { name: 'status', type: 'string', required: false, description: 'فیلتر بر اساس وضعیت', example: 'pending,filled,cancelled' },
        { name: 'symbol', type: 'string', required: false, description: 'فیلتر بر اساس نماد', example: 'AAPL' },
        { name: 'limit', type: 'number', required: false, description: 'تعداد نتایج', example: '50' }
      ],
      headers: [
        { name: 'Authorization', value: 'Bearer {token}', required: true }
      ],
      responseExample: `{
  "orders": [
    {
      "id": "ord_789",
      "symbol": "AAPL",
      "side": "buy",
      "quantity": 10,
      "orderType": "limit",
      "price": 175.00,
      "status": "filled",
      "filledQuantity": 10,
      "averageFillPrice": 174.95,
      "created": "2024-01-20T10:30:00Z",
      "filled": "2024-01-20T10:31:15Z"
    }
  ]
}`
    },
    {
      id: 'orders-create',
      name: 'ثبت سفارش',
      method: 'POST',
      path: '/api/trading/orders',
      description: 'ثبت یک سفارش معاملاتی جدید',
      category: 'Trading',
      requiresAuth: true,
      parameters: [],
      headers: [
        { name: 'Authorization', value: 'Bearer {token}', required: true },
        { name: 'Content-Type', value: 'application/json', required: true }
      ],
      requestBody: `{
  "symbol": "AAPL",
  "side": "buy",
  "quantity": 10,
  "orderType": "limit",
  "price": 175.00,
  "timeInForce": "GTC",
  "stopLoss": 170.00,
  "takeProfit": 185.00
}`,
      responseExample: `{
  "order": {
    "id": "ord_101112",
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 10,
    "orderType": "limit",
    "price": 175.00,
    "status": "pending",
    "created": "2024-01-20T16:00:00Z"
  }
}`
    },
    // Risk Management
    {
      id: 'risk-assessment',
      name: 'ارزیابی ریسک',
      method: 'POST',
      path: '/api/risk/assessment',
      description: 'تحلیل معیارهای ریسک پرتفوی',
      category: 'Risk Management',
      requiresAuth: true,
      parameters: [],
      headers: [
        { name: 'Authorization', value: 'Bearer {token}', required: true },
        { name: 'Content-Type', value: 'application/json', required: true }
      ],
      requestBody: `{
  "portfolioId": "port_123",
  "timeHorizon": "1m",
  "confidenceLevel": 0.95
}`,
      responseExample: `{
  "riskMetrics": {
    "portfolioValue": 50000.00,
    "valueAtRisk": {
      "1day": 1250.00,
      "1week": 2875.00,
      "1month": 6500.00
    },
    "sharpeRatio": 1.45,
    "beta": 1.12,
    "volatility": 0.18,
    "maxDrawdown": 0.08,
    "concentrationRisk": "medium"
  }
}`
    },
    // Analytics
    {
      id: 'analytics-performance',
      name: 'تحلیل عملکرد',
      method: 'GET',
      path: '/api/analytics/performance',
      description: 'دریافت تحلیل جزئی عملکرد',
      category: 'Analytics',
      requiresAuth: true,
      parameters: [
        { name: 'portfolioId', type: 'string', required: true, description: 'شناسه پرتفوی', example: 'port_123' },
        { name: 'period', type: 'string', required: false, description: 'بازه تحلیل', example: '1m,3m,6m,1y' }
      ],
      headers: [
        { name: 'Authorization', value: 'Bearer {token}', required: true }
      ],
      responseExample: `{
  "performance": {
    "totalReturn": 0.0514,
    "annualizedReturn": 0.0892,
    "sharpeRatio": 1.45,
    "maxDrawdown": 0.08,
    "winRate": 0.67,
    "profitFactor": 2.34,
    "monthlyReturns": [0.021, 0.015, 0.018],
    "benchmarkComparison": {
      "sp500": 0.045,
      "outperformance": 0.0064
    }
  }
}`
    }
  ];

  const categories = ['all', ...Array.from(new Set(apiEndpoints.map(endpoint => endpoint.category)))];

  const filteredEndpoints = selectedCategory === 'all'
    ? apiEndpoints
    : apiEndpoints.filter(endpoint => endpoint.category === selectedCategory);

  const addCustomHeader = () => {
    setCustomHeaders(prev => [...prev, { name: '', value: '', required: false }]);
  };

  const removeCustomHeader = (index: number) => {
    setCustomHeaders(prev => prev.filter((_, i) => i !== index));
  };

  const updateCustomHeader = (index: number, field: 'name' | 'value', value: string) => {
    setCustomHeaders(prev => prev.map((header, i) =>
      i === index ? { ...header, [field]: value } : header
    ));
  };

  const executeRequest = async () => {
    if (!selectedEndpoint) return;

    setLoading(true);
    const startTime = Date.now();

    try {
      // Build URL with parameters
      let url = `${baseUrl}${selectedEndpoint.path}`;
      if (selectedEndpoint.method === 'GET' && selectedEndpoint.parameters.length > 0) {
        const params = new URLSearchParams();
        selectedEndpoint.parameters.forEach(param => {
          if (param.example) {
            params.append(param.name, param.example);
          }
        });
        if (params.toString()) {
          url += `?${params.toString()}`;
        }
      }

      // Build headers
      const headers: Record<string, string> = {};
      selectedEndpoint.headers.forEach(header => {
        if (header.name === 'Authorization' && apiKey) {
          headers[header.name] = `Bearer ${apiKey}`;
        } else {
          headers[header.name] = header.value;
        }
      });
      customHeaders.forEach(header => {
        if (header.name && header.value) {
          headers[header.name] = header.value;
        }
      });

      // Build request options
      const options: RequestInit = {
        method: selectedEndpoint.method,
        headers
      };

      if (['POST', 'PUT', 'PATCH'].includes(selectedEndpoint.method)) {
        options.body = requestBody || selectedEndpoint.requestBody || '{}';
      }

      // Simulate API call (in real app, make actual request)
      await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000));

      const endTime = Date.now();
      const responseTime = endTime - startTime;

      // Simulate response
      const mockResponse = selectedEndpoint.responseExample;
      const status = Math.random() > 0.1 ? 200 : 400; // 90% success rate

      setResponse(mockResponse);
      setResponseStatus(status);
      setResponseTime(responseTime);

      // Add to history
      const newRequest: APIRequest = {
        id: `req_${Date.now()}`,
        endpoint: selectedEndpoint,
        timestamp: new Date().toISOString(),
        status,
        responseTime,
        response: mockResponse,
        success: status < 400
      };
      setRequestHistory(prev => [newRequest, ...prev.slice(0, 9)]); // Keep last 10

    } catch (error) {
      setResponse(JSON.stringify({ error: 'Request failed' }, null, 2));
      setResponseStatus(500);
      setResponseTime(Date.now() - startTime);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const formatJsonResponse = (jsonString: string) => {
    try {
      return JSON.stringify(JSON.parse(jsonString), null, 2);
    } catch {
      return jsonString;
    }
  };

  const getMethodColor = (method: string) => {
    switch (method) {
      case 'GET': return 'bg-green-100 text-green-800';
      case 'POST': return 'bg-blue-100 text-blue-800';
      case 'PUT': return 'bg-yellow-100 text-yellow-800';
      case 'DELETE': return 'bg-red-100 text-red-800';
      case 'PATCH': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Market Data': return <TrendingUp className="w-4 h-4" />;
      case 'Portfolio': return <DollarSign className="w-4 h-4" />;
      case 'Trading': return <Activity className="w-4 h-4" />;
      case 'Risk Management': return <Shield className="w-4 h-4" />;
      case 'Analytics': return <BarChart3 className="w-4 h-4" />;
      default: return <Code2 className="w-4 h-4" />;
    }
  };

  useEffect(() => {
    if (selectedEndpoint) {
      setRequestBody(selectedEndpoint.requestBody || '');
      setCustomHeaders([]);
    }
  }, [selectedEndpoint]);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <Terminal className="w-8 h-8 text-blue-600" />
            محیط آزمایش API
          </h1>
          <p className="text-muted-foreground">
            کاوشگر تعاملی API برای تست و اشکال‌زدایی مسیرهای پلتفرم اکتپوس
          </p>
        </div>

        {/* Quick Stats */}
        <div className="flex gap-4">
          <Card className="w-32">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-blue-600">{apiEndpoints.length}</div>
              <div className="text-xs text-gray-500">مسیرها</div>
            </CardContent>
          </Card>
          <Card className="w-32">
            <CardContent className="p-3 text-center">
              <div className="text-2xl font-bold text-green-600">{requestHistory.filter(r => r.success).length}</div>
              <div className="text-xs text-gray-500">موفق</div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            پیکربندی
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="baseUrl">آدرس پایه</Label>
              <div className="flex gap-2">
                <Globe className="w-4 h-4 mt-2 text-gray-500" />
                <Input
                  id="baseUrl"
                  value={baseUrl}
                  onChange={(e) => setBaseUrl(e.target.value)}
                  placeholder="http://localhost:8000"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="apiKey">کلید API</Label>
              <div className="flex gap-2">
                <Key className="w-4 h-4 mt-2 text-gray-500" />
                <Input
                  id="apiKey"
                  type={showApiKey ? 'text' : 'password'}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="کلید API شما"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowApiKey(!showApiKey)}
                >
                  {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-12">
        {/* Endpoints List */}
        <div className="lg:col-span-4">
          <Card className="h-fit">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="w-5 h-5" />
                مسیرهای API
              </CardTitle>
              <div className="flex gap-2 flex-wrap">
                {categories.map(category => (
                  <Button
                    key={category}
                    variant={selectedCategory === category ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setSelectedCategory(category)}
                    className="text-xs"
                  >
                    {category === 'all' ? 'همه' : (CATEGORY_LABELS[category] ?? category)}
                  </Button>
                ))}
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="max-h-96 overflow-y-auto">
                {filteredEndpoints.map(endpoint => (
                  <div
                    key={endpoint.id}
                    className={`p-4 border-b cursor-pointer hover:bg-gray-50 ${
                      selectedEndpoint?.id === endpoint.id ? 'bg-blue-50 border-l-4 border-l-blue-500' : ''
                    }`}
                    onClick={() => setSelectedEndpoint(endpoint)}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Badge className={`text-xs ${getMethodColor(endpoint.method)}`}>
                        {endpoint.method}
                      </Badge>
                      {endpoint.requiresAuth && <Shield className="w-3 h-3 text-orange-500" />}
                      {getCategoryIcon(endpoint.category)}
                    </div>
                    <div className="font-medium text-sm">{endpoint.name}</div>
                    <div className="text-xs text-gray-500 font-mono">{endpoint.path}</div>
                    <div className="text-xs text-gray-600 mt-1">{endpoint.description}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Request Builder */}
        <div className="lg:col-span-8">
          {selectedEndpoint ? (
            <Tabs defaultValue="request" className="space-y-4">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="request">درخواست</TabsTrigger>
                <TabsTrigger value="response">پاسخ</TabsTrigger>
                <TabsTrigger value="docs">مستندات</TabsTrigger>
                <TabsTrigger value="history">تاریخچه</TabsTrigger>
              </TabsList>

              <TabsContent value="request" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Zap className="w-5 h-5" />
                        سازنده درخواست
                      </div>
                      <Button
                        onClick={executeRequest}
                        disabled={loading}
                        className="flex items-center gap-2"
                      >
                        {loading ? (
                          <RotateCcw className="w-4 h-4 animate-spin" />
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                        {loading ? 'در حال اجرا...' : 'اجرا'}
                      </Button>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Endpoint Info */}
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge className={getMethodColor(selectedEndpoint.method)}>
                          {selectedEndpoint.method}
                        </Badge>
                        <code className="text-sm font-mono">{baseUrl}{selectedEndpoint.path}</code>
                      </div>
                      <div className="text-sm text-gray-600">{selectedEndpoint.description}</div>
                    </div>

                    {/* Parameters */}
                    {selectedEndpoint.parameters.length > 0 && (
                      <div>
                        <Label className="text-base font-medium">پارامترها</Label>
                        <div className="space-y-2 mt-2">
                          {selectedEndpoint.parameters.map(param => (
                            <div key={param.name} className="grid grid-cols-4 gap-2 items-center p-2 border rounded">
                              <div>
                                <div className="font-mono text-sm">{param.name}</div>
                                {param.required && <Badge variant="destructive" className="text-xs">اجباری</Badge>}
                              </div>
                              <div className="text-sm text-gray-600">{param.type}</div>
                              <Input
                                placeholder={param.example}
                                className="text-sm"
                              />
                              <div className="text-xs text-gray-500">{param.description}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Headers */}
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <Label className="text-base font-medium">هدرها</Label>
                        <Button variant="outline" size="sm" onClick={addCustomHeader}>
                          <Plus className="w-4 h-4 mr-1" />
                          افزودن هدر
                        </Button>
                      </div>
                      <div className="space-y-2">
                        {selectedEndpoint.headers.map(header => (
                          <div key={header.name} className="grid grid-cols-3 gap-2 items-center p-2 bg-gray-50 rounded">
                            <Input value={header.name} disabled />
                            <Input
                              value={header.name === 'Authorization' && apiKey ? `Bearer ${apiKey}` : header.value}
                              disabled={header.name === 'Authorization'}
                            />
                            {header.required && <Badge variant="destructive" className="text-xs">اجباری</Badge>}
                          </div>
                        ))}
                        {customHeaders.map((header, index) => (
                          <div key={index} className="grid grid-cols-3 gap-2 items-center p-2 border rounded">
                            <Input
                              placeholder="نام هدر"
                              value={header.name}
                              onChange={(e) => updateCustomHeader(index, 'name', e.target.value)}
                            />
                            <Input
                              placeholder="مقدار هدر"
                              value={header.value}
                              onChange={(e) => updateCustomHeader(index, 'value', e.target.value)}
                            />
                            <Button variant="outline" size="sm" onClick={() => removeCustomHeader(index)}>
                              <Minus className="w-4 h-4" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Request Body */}
                    {['POST', 'PUT', 'PATCH'].includes(selectedEndpoint.method) && (
                      <div>
                        <Label className="text-base font-medium">بدنه درخواست</Label>
                        <textarea
                          className="w-full h-32 p-3 border rounded-md font-mono text-sm mt-2"
                          value={requestBody}
                          onChange={(e) => setRequestBody(e.target.value)}
                          placeholder="بدنه درخواست JSON"
                        />
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="response" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Database className="w-5 h-5" />
                        پاسخ
                      </div>
                      <div className="flex items-center gap-2">
                        {responseStatus && (
                          <Badge className={responseStatus < 400 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                            {responseStatus}
                          </Badge>
                        )}
                        {responseTime && (
                          <Badge variant="outline" className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {responseTime}ms
                          </Badge>
                        )}
                        {response && (
                          <Button variant="outline" size="sm" onClick={() => copyToClipboard(response)}>
                            <Copy className="w-4 h-4" />
                          </Button>
                        )}
                      </div>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {response ? (
                      <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
                        {formatJsonResponse(response)}
                      </pre>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        برای مشاهده پاسخ، یک درخواست اجرا کنید
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="docs" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="w-5 h-5" />
                      مستندات
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <h3 className="font-semibold text-lg mb-2">{selectedEndpoint.name}</h3>
                      <p className="text-gray-600 mb-4">{selectedEndpoint.description}</p>

                      <div className="grid gap-4">
                        <div>
                          <h4 className="font-medium mb-2">مسیر</h4>
                          <code className="bg-gray-100 p-2 rounded block">
                            {selectedEndpoint.method} {selectedEndpoint.path}
                          </code>
                        </div>

                        {selectedEndpoint.parameters.length > 0 && (
                          <div>
                            <h4 className="font-medium mb-2">پارامترها</h4>
                            <div className="space-y-2">
                              {selectedEndpoint.parameters.map(param => (
                                <div key={param.name} className="border-l-4 border-blue-500 pl-4">
                                  <div className="flex items-center gap-2">
                                    <code className="font-mono text-sm">{param.name}</code>
                                    <Badge variant={param.required ? 'destructive' : 'secondary'} className="text-xs">
                                      {param.required ? 'اجباری' : 'اختیاری'}
                                    </Badge>
                                    <Badge variant="outline" className="text-xs">{param.type}</Badge>
                                  </div>
                                  <p className="text-sm text-gray-600 mt-1">{param.description}</p>
                                  <p className="text-xs text-gray-500 mt-1">مثال: <code>{param.example}</code></p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        <div>
                          <h4 className="font-medium mb-2">نمونه پاسخ</h4>
                          <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
                            {formatJsonResponse(selectedEndpoint.responseExample)}
                          </pre>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="history" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <History className="w-5 h-5" />
                      تاریخچه درخواست‌ها
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {requestHistory.length > 0 ? (
                      <div className="space-y-3">
                        {requestHistory.map(request => (
                          <div key={request.id} className="border rounded-lg p-3">
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <Badge className={getMethodColor(request.endpoint.method)}>
                                  {request.endpoint.method}
                                </Badge>
                                <span className="font-mono text-sm">{request.endpoint.path}</span>
                              </div>
                              <div className="flex items-center gap-2">
                                {request.success ? (
                                  <CheckCircle className="w-4 h-4 text-green-500" />
                                ) : (
                                  <XCircle className="w-4 h-4 text-red-500" />
                                )}
                                <Badge className={request.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                                  {request.status}
                                </Badge>
                                <Badge variant="outline" className="flex items-center gap-1">
                                  <Clock className="w-3 h-3" />
                                  {request.responseTime}ms
                                </Badge>
                              </div>
                            </div>
                            <div className="text-xs text-gray-500">
                              {new Date(request.timestamp).toLocaleString('fa-IR')}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        هنوز درخواستی ثبت نشده است. برای مشاهده تاریخچه، چند درخواست API اجرا کنید.
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <Code2 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">یک مسیر API را انتخاب کنید</h3>
                <p className="text-gray-600">
                  یک مسیر را از فهرست انتخاب کنید تا تست و کاوش API را شروع کنید
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}