"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { FundamentalAnalysis } from '@/components/fundamental/fundamental-analysis';
import { 
  Search, 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  RefreshCw,
  FileText,
  Download,
  Brain,
  Zap,
  Target,
  Eye,
  BarChart3,
  PieChart,
  LineChart,
  AlertTriangle,
  CheckCircle,
  Clock,
  Layers,
  Database,
  Globe,
  Shield
} from 'lucide-react';

const PREDEFINED_ASSETS = [
  { symbol: 'BTC-USD', name: 'بیت‌کوین', type: 'crypto', category: 'لایه ۱', marketCap: '$831B' },
  { symbol: 'ETH-USD', name: 'اتریوم', type: 'crypto', category: 'قرارداد هوشمند', marketCap: '$310B' },
  { symbol: 'AAPL', name: 'اپل', type: 'stock', category: 'فناوری', marketCap: '$2.8T' },
  { symbol: 'TSLA', name: 'تسلا', type: 'stock', category: 'خودرو برقی/انرژی', marketCap: '$780B' },
  { symbol: 'NVDA', name: 'انویدیا', type: 'stock', category: 'هوش مصنوعی/نیمه‌هادی', marketCap: '$1.2T' },
  { symbol: 'MSFT', name: 'مایکروسافت', type: 'stock', category: 'ابر/نرم‌افزار', marketCap: '$2.9T' },
  { symbol: 'GOOGL', name: 'آلفابت', type: 'stock', category: 'فناوری/هوش مصنوعی', marketCap: '$1.7T' },
  { symbol: 'LINK-USD', name: 'چین‌لینک', type: 'crypto', category: 'شبکه اوراکل', marketCap: '$8.2B' },
  { symbol: 'SOL-USD', name: 'سولانا', type: 'crypto', category: 'لایه ۱', marketCap: '$43B' },
  { symbol: 'AVAX-USD', name: 'آوالانچ', type: 'crypto', category: 'لایه ۱', marketCap: '$15B' },
  { symbol: 'MATIC-USD', name: 'پالیگان', type: 'crypto', category: 'لایه ۲', marketCap: '$9.8B' },
  { symbol: 'UNI-USD', name: 'یونی‌سواپ', type: 'crypto', category: 'دیفای', marketCap: '$5.2B' },
];

const NEW_PROJECTS = [
  { symbol: 'ARB-USD', name: 'آربیتروم', type: 'crypto', category: 'لایه ۲', status: 'نوظهور', risk: 'بالا' },
  { symbol: 'OP-USD', name: 'اپتیمیزم', type: 'crypto', category: 'لایه ۲', status: 'در حال رشد', risk: 'متوسط' },
  { symbol: 'BLUR-USD', name: 'بلور', type: 'crypto', category: 'NFT/گیمینگ', status: 'جدید', risk: 'بسیار بالا' },
  { symbol: 'LDO-USD', name: 'لیدو DAO', type: 'crypto', category: 'استیکینگ نقدشونده', status: 'تثبیت‌شده', risk: 'متوسط' },
  { symbol: 'RDNT-USD', name: 'رادیانت کپیتال', type: 'crypto', category: 'دیفای/وام‌دهی', status: 'نوظهور', risk: 'بالا' },
  { symbol: 'GMX-USD', name: 'جی‌ام‌ایکس', type: 'crypto', category: 'دیفای/مشتقات', status: 'در حال رشد', risk: 'متوسط' },
];

export default function FundamentalDataPage() {
  const [selectedAsset, setSelectedAsset] = useState('BTC-USD');
  const [searchTerm, setSearchTerm] = useState('');
  const [dashboardData, setDashboardData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [reportGenerating, setReportGenerating] = useState(false);
  const [deepAnalysisLoading, setDeepAnalysisLoading] = useState(false);
  const [aiInsights, setAiInsights] = useState<any>(null);
  const [selectedCategory, setSelectedCategory] = useState('all');

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const response = await fetch('/api/fundamental/dashboard');
        const data = await response.json();
        setDashboardData(data);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const handleAssetSelect = (symbol: string) => {
    setSelectedAsset(symbol);
  };

  const handleSearch = () => {
    if (searchTerm.trim()) {
      setSelectedAsset(searchTerm.trim().toUpperCase());
    }
  };

  const generateDeepResearchReport = async () => {
    setReportGenerating(true);
    try {
      // Generate AI-powered research report
      const response = await fetch('/api/llm/reports/generate-insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          asset: selectedAsset,
          analysis_type: 'deep_fundamental',
          include_pdf: true
        })
      });
      
      const result = await response.json();
      
      // Simulate PDF generation
      const pdfBlob = new Blob([`
        DEEP FUNDAMENTAL ANALYSIS REPORT
        Asset: ${selectedAsset}
        Generated: ${new Date().toISOString()}
        
        ${result.raw_ai_response || 'Comprehensive analysis completed'}
        
        This report contains detailed fundamental analysis including:
        - Financial metrics and ratios
        - Competitive landscape analysis
        - Market position assessment
        - Risk factors and opportunities
        - AI-powered insights and recommendations
      `], { type: 'application/pdf' });
      
      const url = URL.createObjectURL(pdfBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedAsset}_fundamental_analysis_${new Date().getTime()}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('Error generating report:', error);
    } finally {
      setReportGenerating(false);
    }
  };

  const runDeepAnalysis = async () => {
    setDeepAnalysisLoading(true);
    try {
      const response = await fetch('/api/llm/reports/generate-insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const insights = await response.json();
      setAiInsights(insights);
    } catch (error) {
      console.error('Error running deep analysis:', error);
    } finally {
      setDeepAnalysisLoading(false);
    }
  };

  const filteredAssets = PREDEFINED_ASSETS.filter(asset => {
    const matchesSearch = asset.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         asset.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || asset.category.toLowerCase().includes(selectedCategory.toLowerCase());
    return matchesSearch && matchesCategory;
  });

  const filteredNewProjects = NEW_PROJECTS.filter(project => {
    const matchesSearch = project.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         project.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || project.category.toLowerCase().includes(selectedCategory.toLowerCase());
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">تحقیق بنیادی</h1>
          <p className="text-muted-foreground mt-1">
            تحلیل و تحقیق بنیادی — داشبورد از API در صورت در دسترس بودن
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <Button onClick={runDeepAnalysis} disabled={deepAnalysisLoading} variant="outline" size="sm">
            <Brain className="h-4 w-4 mr-2" />
            {deepAnalysisLoading ? 'در حال تحلیل...' : 'تحلیل هوش مصنوعی'}
          </Button>
          <Button onClick={generateDeepResearchReport} disabled={reportGenerating} variant="secondary" size="sm">
            <FileText className="h-4 w-4 mr-2" />
            {reportGenerating ? 'در حال تولید...' : 'تولید PDF'}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="research" className="w-full">
        <div className="border-b border-border/50 bg-card/30 px-4 py-2 rounded-lg">
          <TabsList className="grid w-full max-w-2xl grid-cols-4">
            <TabsTrigger value="research">تحقیق عمیق</TabsTrigger>
            <TabsTrigger value="new-projects">پروژه‌های جدید</TabsTrigger>
            <TabsTrigger value="ai-insights">بینش‌های هوش مصنوعی</TabsTrigger>
            <TabsTrigger value="reports">گزارش‌ها</TabsTrigger>
          </TabsList>
        </div>

          <TabsContent value="research" className="mt-6 space-y-6">
            {/* Asset Selection & Research Tools */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Database className="h-5 w-5 text-foreground" />
                  <span>مرکز تحقیق</span>
                </CardTitle>
                <CardDescription>انتخاب دارایی برای تحلیل بنیادی عمیق با بینش‌های هوش مصنوعی</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">انتخاب دارایی</label>
                    <Select value={selectedAsset} onValueChange={handleAssetSelect}>
                      <SelectTrigger>
                        <SelectValue placeholder="یک دارایی انتخاب کنید" />
                      </SelectTrigger>
                      <SelectContent>
                        {PREDEFINED_ASSETS.map((asset) => (
                          <SelectItem key={asset.symbol} value={asset.symbol}>
                            <div className="flex items-center space-x-2">
                              <span className="font-medium">{asset.symbol}</span>
                              <span className="text-muted-foreground">- {asset.name}</span>
                              <Badge variant="outline" className="text-xs">
                                {asset.category}
                              </Badge>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">فیلتر دسته</label>
                    <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                      <SelectTrigger>
                        <SelectValue placeholder="همه دسته‌ها" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">همه دسته‌ها</SelectItem>
                        <SelectItem value="لایه ۱">لایه ۱</SelectItem>
                        <SelectItem value="لایه ۲">لایه ۲</SelectItem>
                        <SelectItem value="دیفای">دیفای</SelectItem>
                        <SelectItem value="فناوری">فناوری</SelectItem>
                        <SelectItem value="هوش مصنوعی">هوش مصنوعی</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">جستجوی دارایی</label>
                    <div className="flex space-x-2">
                      <Input
                        placeholder="جستجوی نماد یا نام..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                      />
                      <Button onClick={handleSearch} size="icon" className="btn-morphic">
                        <Search className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>

                {/* Asset Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {filteredAssets.map((asset) => (
                    <div 
                      key={asset.symbol} 
                      className={`neomorphic p-4 rounded-lg cursor-pointer transition-all hover:scale-105 ${
                        selectedAsset === asset.symbol ? 'ring-2 ring-blue-400' : ''
                      }`}
                      onClick={() => handleAssetSelect(asset.symbol)}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <h3 className="font-bold text-lg">{asset.symbol}</h3>
                          <p className="text-sm text-gray-400">{asset.name}</p>
                        </div>
                        <Badge className={`${
                          asset.type === 'crypto' ? 'bg-orange-500/20 text-orange-300' :
                          asset.type === 'stock' ? 'bg-green-500/20 text-green-300' :
                          'bg-blue-500/20 text-blue-300'
                        }`}>
                          {asset.type === 'crypto' ? 'ارز دیجیتال' : asset.type === 'stock' ? 'سهام' : asset.type}
                        </Badge>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">دسته:</span>
                          <span className="text-sm font-medium">{asset.category}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">ارزش بازار:</span>
                          <span className="text-sm font-medium">{asset.marketCap}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Selected Asset Analysis */}
            <FundamentalAnalysis symbol={selectedAsset} />
          </TabsContent>

          <TabsContent value="new-projects" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Zap className="h-5 w-5 text-yellow-400" />
                  <span>پروژه‌های جدید و نوظهور</span>
                </CardTitle>
                <CardDescription>تحقیق و تحلیل پروژه‌ها و توکن‌های جدید بلاک‌چین</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {filteredNewProjects.map((project) => (
                    <div key={project.symbol} className="neomorphic p-4 rounded-lg">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h3 className="font-bold">{project.symbol}</h3>
                          <p className="text-sm text-gray-400">{project.name}</p>
                        </div>
                        <Badge className={`${
                          project.risk === 'بسیار بالا' ? 'bg-red-500/20 text-red-300' :
                          project.risk === 'بالا' ? 'bg-orange-500/20 text-orange-300' :
                          'bg-yellow-500/20 text-yellow-300'
                        }`}>
                          {project.risk}
                        </Badge>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">دسته:</span>
                          <span className="text-sm font-medium">{project.category}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">وضعیت:</span>
                          <Badge className={`text-xs ${
                            project.status === 'تثبیت‌شده' ? 'bg-green-500/20 text-green-300' :
                            project.status === 'در حال رشد' ? 'bg-blue-500/20 text-blue-300' :
                            'bg-purple-500/20 text-purple-300'
                          }`}>
                            {project.status}
                          </Badge>
                        </div>
                      </div>
                      <Button 
                        className="w-full mt-3 btn-morphic text-sm"
                        onClick={() => handleAssetSelect(project.symbol)}
                      >
                        تحقیق درباره پروژه
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="ai-insights" className="space-y-6">
            {aiInsights ? (
              <div className="space-y-6">
                {/* AI Insights Overview */}
                <Card className="glass-card">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Brain className="h-5 w-5 text-purple-400" />
                      <span>هوشمندی بازار با هوش مصنوعی</span>
                    </CardTitle>
                    <CardDescription>بینش‌های مبتنی بر LLM از تحلیل جامع بازار</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                      <div className="text-center p-4 neomorphic rounded-lg">
                        <div className="text-2xl font-bold text-green-400">{aiInsights.market_summary?.bullish_signals || 0}</div>
                        <div className="text-sm text-gray-400">سیگنال‌های صعودی</div>
                      </div>
                      <div className="text-center p-4 neomorphic rounded-lg">
                        <div className="text-2xl font-bold text-red-400">{aiInsights.market_summary?.bearish_signals || 0}</div>
                        <div className="text-sm text-gray-400">سیگنال‌های نزولی</div>
                      </div>
                      <div className="text-center p-4 neomorphic rounded-lg">
                        <div className="text-2xl font-bold text-blue-400">{aiInsights.market_summary?.total_assets_analyzed || 0}</div>
                        <div className="text-sm text-gray-400">دارایی‌های تحلیل‌شده</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* AI Generated Insights */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {aiInsights.insights?.map((insight: any, index: number) => (
                    <Card key={insight.id} className="glass-card">
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-lg">{insight.title}</CardTitle>
                          <Badge className={`${
                            insight.type === 'bullish' ? 'bg-green-500/20 text-green-300' :
                            insight.type === 'bearish' ? 'bg-red-500/20 text-red-300' :
                            insight.type === 'warning' ? 'bg-orange-500/20 text-orange-300' :
                            'bg-blue-500/20 text-blue-300'
                          }`}>
                            {insight.type}
                          </Badge>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <p className="text-gray-300 mb-4">{insight.summary}</p>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-400">اطمینان:</span>
                            <div className="flex items-center space-x-2">
                              <Progress value={insight.confidence} className="w-20 h-2" />
                              <span className="text-sm font-medium">{insight.confidence}%</span>
                            </div>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-400">تأثیر:</span>
                            <Badge className={`text-xs ${
                              insight.impact === 'high' ? 'bg-red-500/20 text-red-300' :
                              insight.impact === 'medium' ? 'bg-yellow-500/20 text-yellow-300' :
                              'bg-green-500/20 text-green-300'
                            }`}>
                              {insight.impact}
                            </Badge>
                          </div>
                        </div>
                        {insight.ai_analysis && (
                          <div className="mt-4 p-3 bg-gray-800/50 rounded-lg">
                            <p className="text-sm text-gray-300">{insight.ai_analysis}</p>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            ) : (
              <Card className="glass-card">
                <CardContent className="text-center py-12">
                  <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium mb-2">هنوز بینشی تولید نشده است</h3>
                  <p className="text-gray-400 mb-4">برای تولید بینش‌های جامع بازار روی «تحلیل هوش مصنوعی» کلیک کنید</p>
                  <Button onClick={runDeepAnalysis} disabled={deepAnalysisLoading} className="btn-morphic">
                    <Brain className="h-4 w-4 mr-2" />
                    {deepAnalysisLoading ? 'در حال تحلیل...' : 'تولید بینش هوش مصنوعی'}
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="reports" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="h-5 w-5 text-green-400" />
                  <span>گزارش‌های تحقیقاتی</span>
                </CardTitle>
                <CardDescription>تولید گزارش‌های جامع PDF با استفاده از تحلیل هوش مصنوعی</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h3 className="font-semibold">انواع گزارش</h3>
                    <div className="space-y-3">
                      <div className="p-4 neomorphic rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium">تحلیل بنیادی عمیق</h4>
                          <Badge className="bg-blue-500/20 text-blue-300">جامع</Badge>
                        </div>
                        <p className="text-sm text-gray-400 mb-3">تحلیل بنیادی کامل همراه با بینش‌های هوش مصنوعی، متریک‌های مالی و توصیه‌ها</p>
                        <Button
                          onClick={generateDeepResearchReport}
                          disabled={reportGenerating}
                          className="w-full btn-morphic"
                        >
                          <Download className="h-4 w-4 mr-2" />
                          {reportGenerating ? 'در حال تولید...' : 'تولید PDF'}
                        </Button>
                      </div>

                      <div className="p-4 neomorphic rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium">تکنیکال + بنیادی</h4>
                          <Badge className="bg-purple-500/20 text-purple-300">ترکیبی</Badge>
                        </div>
                        <p className="text-sm text-gray-400 mb-3">تحلیل یکپارچه تکنیکال و بنیادی برای دید کامل بازار</p>
                        <Button className="w-full btn-morphic" variant="outline">
                          <Download className="h-4 w-4 mr-2" />
                          به‌زودی
                        </Button>
                      </div>

                      <div className="p-4 neomorphic rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium">تحلیل پرتفوی</h4>
                          <Badge className="bg-green-500/20 text-green-300">چند-دارایی</Badge>
                        </div>
                        <p className="text-sm text-gray-400 mb-3">تحلیل کامل پرتفوی همراه با ارزیابی ریسک و بهینه‌سازی</p>
                        <Button className="w-full btn-morphic" variant="outline">
                          <Download className="h-4 w-4 mr-2" />
                          به‌زودی
                        </Button>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h3 className="font-semibold">امکانات گزارش</h3>
                    <div className="space-y-3">
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-400" />
                        <span className="text-sm">بینش‌ها و تحلیل مبتنی بر هوش مصنوعی</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-400" />
                        <span className="text-sm">یکپارچگی با داده بازار بلادرنگ</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-400" />
                        <span className="text-sm">قالب‌بندی حرفه‌ای PDF</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-400" />
                        <span className="text-sm">تحقیق در سطح نهادی</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-400" />
                        <span className="text-sm">ارزیابی ریسک و توصیه‌ها</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-400" />
                        <span className="text-sm">تحلیل فضای رقابتی</span>
                      </div>
                    </div>

                    <div className="mt-6 p-4 bg-indigo-500/10 border border-indigo-500/20 rounded-lg">
                      <h4 className="font-medium text-indigo-300 mb-2">تحقیق مبتنی بر هوش مصنوعی</h4>
                      <p className="text-sm text-gray-300">
                        گزارش‌های ما از مدل‌های پیشرفته LLM برای تحلیل داده بازار،
                        احساسات اخبار و متریک‌های بنیادی استفاده می‌کنند تا تحقیق و
                        بینش‌های عملیاتی در سطح نهادی ارائه دهند.
                      </p>
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