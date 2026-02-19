'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { toast } from '@/components/ui/toast';
import {
  Brain,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Target,
  Zap,
  Download,
  RefreshCw,
  Eye,
  BarChart3,
  PieChart,
  Activity,
  DollarSign,
  Shield,
  Globe,
  Lightbulb,
  Clock,
  CheckCircle,
  XCircle,
} from 'lucide-react';

interface AIInsight {
  id: string;
  type: 'bullish' | 'bearish' | 'neutral' | 'warning' | 'opportunity';
  title: string;
  summary: string;
  confidence: number;
  impact: 'high' | 'medium' | 'low';
  sources: string[];
  timestamp: string;
}

interface ReportSection {
  id: string;
  name: string;
  status: 'completed' | 'processing' | 'pending' | 'failed';
  progress: number;
  insights: number;
  lastUpdated: string;
}

interface DataSource {
  id: string;
  name: string;
  type: 'market_data' | 'news' | 'social' | 'technical' | 'fundamental' | 'sentiment';
  status: 'active' | 'inactive' | 'error';
  lastSync: string;
  recordsProcessed: number;
}

const REPORT_TYPES = [
  { value: 'market_summary', label: 'Market Summary' },
  { value: 'risk_assessment', label: 'Risk Assessment' },
  { value: 'sentiment', label: 'Sentiment Analysis' },
  { value: 'technical', label: 'Technical Analysis' },
] as const;

/** Extract report narrative from backend response (multiple possible keys). */
function getReportTextFromResponse(data: Record<string, unknown>): string | null {
  const raw = data.raw_ai_response ?? data.report ?? data.content;
  if (typeof raw === 'string' && raw.trim()) return raw.trim();
  const insights = data.insights as Array<{ ai_analysis?: string; summary?: string }> | undefined;
  if (Array.isArray(insights)) {
    const parts = insights
      .map((i) => (i?.ai_analysis ?? i?.summary) as string | undefined)
      .filter((s): s is string => typeof s === 'string' && s.length > 0);
    if (parts.length) return parts.join('\n\n');
  }
  return null;
}

export function ReportsContent() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState('7d');
  const [reportType, setReportType] = useState<string>('market_summary');
  const [activeTab, setActiveTab] = useState('insights');
  const [lastReportText, setLastReportText] = useState<string | null>(null);
  const [reportModelUsed, setReportModelUsed] = useState<string | null>(null);
  const [reportGeneratedAt, setReportGeneratedAt] = useState<string | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [aiInsights, setAiInsights] = useState<AIInsight[]>([
    {
      id: '1',
      type: 'bullish',
      title: 'Strong Momentum in Tech Sector',
      summary: 'AI analysis indicates sustained upward pressure in NVDA, MSFT, and GOOGL based on earnings momentum, institutional flow, and technical patterns. Expected 12-15% upside over next 30 days.',
      confidence: 87,
      impact: 'high',
      sources: ['Technical Analysis', 'Market Data', 'News Sentiment', 'Options Flow'],
      timestamp: new Date().toISOString()
    },
    {
      id: '2',
      type: 'warning',
      title: 'Crypto Volatility Alert',
      summary: 'Unusual options activity and social sentiment divergence detected in BTC-USD and ETH-USD. Potential 20-25% volatility spike expected within 48-72 hours.',
      confidence: 78,
      impact: 'medium',
      sources: ['Social Sentiment', 'Options Data', 'Technical Indicators'],
      timestamp: new Date().toISOString()
    },
    {
      id: '3',
      type: 'opportunity',
      title: 'Commodities Rotation Signal',
      summary: 'Inflation hedge rotation detected. GLD and SLV showing strong accumulation patterns while institutional money rotates from growth to value. 8-12% upside potential.',
      confidence: 82,
      impact: 'medium',
      sources: ['Institutional Flow', 'Macro Analysis', 'Technical Patterns'],
      timestamp: new Date().toISOString()
    },
    {
      id: '4',
      type: 'neutral',
      title: 'Stablecoin Stability Confirmed',
      summary: 'USDT and USDC maintaining healthy peg stability. No depegging risks detected. Suitable for portfolio stability and cash management strategies.',
      confidence: 95,
      impact: 'low',
      sources: ['On-chain Data', 'Market Data', 'Liquidity Analysis'],
      timestamp: new Date().toISOString()
    }
  ]);

  const [reportSections, setReportSections] = useState<ReportSection[]>([
    { id: 'market', name: 'Market Analysis', status: 'completed', progress: 100, insights: 8, lastUpdated: '2 min ago' },
    { id: 'portfolio', name: 'Portfolio Performance', status: 'completed', progress: 100, insights: 12, lastUpdated: '5 min ago' },
    { id: 'risk', name: 'Risk Assessment', status: 'processing', progress: 75, insights: 6, lastUpdated: '1 min ago' },
    { id: 'sentiment', name: 'Sentiment Analysis', status: 'completed', progress: 100, insights: 15, lastUpdated: '3 min ago' },
    { id: 'technical', name: 'Technical Signals', status: 'processing', progress: 60, insights: 9, lastUpdated: 'Now' },
    { id: 'macro', name: 'Macro Environment', status: 'pending', progress: 0, insights: 0, lastUpdated: 'Pending' }
  ]);

  const [dataSources, setDataSources] = useState<DataSource[]>([
    { id: 'market', name: 'Market Data Feed', type: 'market_data', status: 'active', lastSync: '30s ago', recordsProcessed: 15420 },
    { id: 'news', name: 'News Analytics', type: 'news', status: 'active', lastSync: '1m ago', recordsProcessed: 2847 },
    { id: 'social', name: 'Social Sentiment', type: 'social', status: 'active', lastSync: '45s ago', recordsProcessed: 8932 },
    { id: 'technical', name: 'Technical Indicators', type: 'technical', status: 'active', lastSync: '15s ago', recordsProcessed: 5621 },
    { id: 'fundamental', name: 'Fundamental Data', type: 'fundamental', status: 'active', lastSync: '2m ago', recordsProcessed: 1205 },
    { id: 'sentiment', name: 'Sentiment Engine', type: 'sentiment', status: 'active', lastSync: '20s ago', recordsProcessed: 3847 }
  ]);

  // Load data sources from backend
  useEffect(() => {
    const loadDataSources = async () => {
      try {
        const response = await fetch('/api/llm/reports/data-sources');
        if (response.ok) {
          const data = await response.json();
          if (data.sources) {
            setDataSources(data.sources);
          }
        }
      } catch (error) {
        console.error('Error loading data sources:', error);
      }
    };

    const loadAnalysisStatus = async () => {
      try {
        const response = await fetch('/api/llm/reports/analysis-status');
        if (response.ok) {
          const data = await response.json();
          if (data.sections) {
            setReportSections(data.sections);
          }
        }
      } catch (error) {
        console.error('Error loading analysis status:', error);
      }
    };

    loadDataSources();
    loadAnalysisStatus();
    
    // Set up periodic refresh
    const interval = setInterval(() => {
      loadDataSources();
      loadAnalysisStatus();
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const generateAIReport = async () => {
    setIsGenerating(true);
    setGenerateError(null);

    try {
      const response = await fetch('/api/llm/reports/generate-insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report_type: reportType }),
      });

      const data = await response.json().catch(() => ({}));

      if (response.ok) {
        const dataObj = data as Record<string, unknown>;
        if (dataObj.insights) setAiInsights(dataObj.insights as AIInsight[]);
        setReportSections((prev) =>
          prev.map((section) => ({
            ...section,
            progress: 100,
            status: 'completed',
            lastUpdated: 'Just now',
          }))
        );

        const reportText = getReportTextFromResponse(dataObj);
        if (reportText) {
          setLastReportText(reportText);
          const marketSummary = dataObj.market_summary as { ai_model?: string } | undefined;
          setReportModelUsed(marketSummary?.ai_model ?? (dataObj.model_used as string) ?? 'AI');
          setReportGeneratedAt(new Date().toISOString());
          setActiveTab('reports');
        }
        toast({
          title: 'Report generated',
          description: reportText ? 'AI report is ready in the Generated Reports tab.' : 'Insights updated.',
          type: 'success',
          duration: 4500,
        });
      } else {
        const message = (data as { error?: string }).error ?? `Request failed (${response.status})`;
        setGenerateError(message);
        toast({
          title: 'Generate report failed',
          description: message,
          type: 'error',
          duration: 6000,
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Network or server error. Is the backend running at BACKEND_URL?';
      setGenerateError(message);
      toast({
        title: 'Generate report failed',
        description: message,
        type: 'error',
        duration: 6000,
      });
    }

    setIsGenerating(false);
  };

  const downloadReport = () => {
    if (!lastReportText || !reportGeneratedAt) return;
    const title = REPORT_TYPES.find((r) => r.value === reportType)?.label ?? 'Report';
    const blob = new Blob(
      [`# ${title}\n\nGenerated: ${new Date(reportGeneratedAt).toLocaleString()}\nModel: ${reportModelUsed ?? 'AI'}\n\n---\n\n${lastReportText}`],
      { type: 'text/markdown' }
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `findash-report-${reportType}-${reportGeneratedAt.slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'bullish': return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'bearish': return <TrendingDown className="w-4 h-4 text-red-500" />;
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'opportunity': return <Target className="w-4 h-4 text-blue-500" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const getInsightColor = (type: string) => {
    switch (type) {
      case 'bullish': return 'bg-green-500/10 text-green-700 border-green-200';
      case 'bearish': return 'bg-red-500/10 text-red-700 border-red-200';
      case 'warning': return 'bg-yellow-500/10 text-yellow-700 border-yellow-200';
      case 'opportunity': return 'bg-blue-500/10 text-blue-700 border-blue-200';
      default: return 'bg-gray-500/10 text-gray-700 border-gray-200';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'processing': return <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-500" />;
      default: return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* AI Report Header */}
      <Card className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border-border">
        <CardHeader>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <Brain className="w-8 h-8 text-primary" />
              <div>
                <CardTitle className="text-2xl">AI Intelligence Report</CardTitle>
                <p className="text-muted-foreground text-sm">
                  Reports are generated by AI models (Llama/FinGPT when configured, otherwise simulated). Choose type and generate.
                </p>
              </div>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <Select value={reportType} onValueChange={setReportType}>
                <SelectTrigger className="w-full sm:w-[180px]">
                  <SelectValue placeholder="Report type" />
                </SelectTrigger>
                <SelectContent>
                  {REPORT_TYPES.map((r) => (
                    <SelectItem key={r.value} value={r.value}>
                      {r.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <div className="flex gap-2">
                {lastReportText && (
                  <Button variant="outline" size="sm" onClick={downloadReport}>
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                )}
                <Button onClick={generateAIReport} disabled={isGenerating}>
                  {isGenerating ? (
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Zap className="w-4 h-4 mr-2" />
                  )}
                  {isGenerating ? 'Generating…' : 'Generate Report'}
                </Button>
              </div>
            </div>
          </div>
          {reportModelUsed && reportGeneratedAt && (
            <p className="text-xs text-muted-foreground mt-2">
              Last report: generated with <span className="font-medium">{reportModelUsed}</span> at{' '}
              {new Date(reportGeneratedAt).toLocaleString()}
            </p>
          )}
          {generateError && (
            <div className="mt-3 flex items-center gap-2 rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>{generateError}</span>
            </div>
          )}
        </CardHeader>
      </Card>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
          <TabsTrigger value="analysis">Analysis Status</TabsTrigger>
          <TabsTrigger value="sources">Data Sources</TabsTrigger>
          <TabsTrigger value="reports">Generated Reports</TabsTrigger>
        </TabsList>

        <TabsContent value="insights" className="space-y-6">
          {/* Key Metrics */}
          <div className="grid gap-4 md:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Insights</CardTitle>
                <Lightbulb className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{aiInsights.length}</div>
                <p className="text-xs text-muted-foreground">
                  AI-generated insights
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Avg Confidence</CardTitle>
                <Target className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {Math.round(aiInsights.reduce((acc, insight) => acc + insight.confidence, 0) / aiInsights.length)}%
                </div>
                <p className="text-xs text-muted-foreground">
                  AI confidence level
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">High Impact</CardTitle>
                <AlertTriangle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {aiInsights.filter(insight => insight.impact === 'high').length}
                </div>
                <p className="text-xs text-muted-foreground">
                  Critical insights
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Data Sources</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{dataSources.length}</div>
                <p className="text-xs text-muted-foreground">
                  Active data feeds
                </p>
              </CardContent>
            </Card>
          </div>

          {/* AI Insights */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="w-5 h-5" />
                Latest AI Insights
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {aiInsights.map((insight) => (
                  <div key={insight.id} className={`p-4 rounded-lg border ${getInsightColor(insight.type)}`}>
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-2">
                        {getInsightIcon(insight.type)}
                        <h3 className="font-semibold">{insight.title}</h3>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                          {insight.confidence}% confidence
                        </Badge>
                        <Badge variant={insight.impact === 'high' ? 'destructive' : insight.impact === 'medium' ? 'default' : 'secondary'}>
                          {insight.impact} impact
                        </Badge>
                      </div>
                    </div>
                    
                    <p className="text-sm mb-3">{insight.summary}</p>
                    
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <span>Sources:</span>
                        <span className="font-medium">{insight.sources.join(', ')}</span>
                      </div>
                      <span>{new Date(insight.timestamp).toLocaleTimeString()}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-6">
          {/* Analysis Progress */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                Analysis Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {reportSections.map((section) => (
                  <div key={section.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(section.status)}
                      <div>
                        <h3 className="font-medium">{section.name}</h3>
                        <p className="text-sm text-muted-foreground">
                          {section.insights} insights • Updated {section.lastUpdated}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-32">
                        <Progress value={section.progress} className="h-2" />
                      </div>
                      <span className="text-sm font-medium w-12">{section.progress}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sources" className="space-y-6">
          {/* Data Sources */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Data Source Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                {dataSources.map((source) => (
                  <div key={source.id} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium">{source.name}</h3>
                      <Badge variant={source.status === 'active' ? 'default' : 'destructive'}>
                        {source.status}
                      </Badge>
                    </div>
                    <div className="space-y-2 text-sm text-muted-foreground">
                      <div className="flex justify-between">
                        <span>Type:</span>
                        <span className="font-medium capitalize">{source.type.replace('_', ' ')}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Last Sync:</span>
                        <span className="font-medium">{source.lastSync}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Records:</span>
                        <span className="font-medium">{source.recordsProcessed.toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reports" className="space-y-6">
          {/* Latest AI-generated report */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <PieChart className="w-5 h-5" />
                Latest AI-Generated Report
              </CardTitle>
              {lastReportText && (
                <Button size="sm" onClick={downloadReport}>
                  <Download className="w-4 h-4 mr-2" />
                  Download (.md)
                </Button>
              )}
            </CardHeader>
            <CardContent>
              {lastReportText ? (
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                    <Badge variant="secondary">{reportModelUsed ?? 'AI'}</Badge>
                    {reportGeneratedAt && (
                      <span>{new Date(reportGeneratedAt).toLocaleString()}</span>
                    )}
                    <span>
                      {REPORT_TYPES.find((r) => r.value === reportType)?.label ?? reportType}
                    </span>
                  </div>
                  <div className="rounded-lg border bg-muted/30 p-4 max-h-[420px] overflow-y-auto whitespace-pre-wrap text-sm">
                    {lastReportText}
                  </div>
                </div>
              ) : (
                <p className="text-muted-foreground text-sm">
                  Generate a report using the &quot;Generate Report&quot; button above. The full AI-written narrative (Llama/FinGPT when configured) will appear here.
                </p>
              )}
            </CardContent>
          </Card>

          {/* Report type templates */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <PieChart className="w-5 h-5" />
                Report Types
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                {REPORT_TYPES.map((r) => (
                  <div key={r.value} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold">{r.label}</h3>
                      <Badge variant={reportType === r.value ? 'default' : 'outline'}>
                        {reportType === r.value ? 'Selected' : 'Select above'}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {r.value === 'market_summary' && 'One-page market analysis: indices, tech, macro, outlook.'}
                      {r.value === 'risk_assessment' && 'Portfolio risk: volatility, correlations, tail risk, recommendations.'}
                      {r.value === 'sentiment' && 'Social and news sentiment, sector highlights, divergences.'}
                      {r.value === 'technical' && 'Support/resistance, momentum, actionable technical signals.'}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 