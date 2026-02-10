'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
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
  XCircle
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

export function ReportsContent() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState('7d');
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
    
    try {
      // Call the backend API to generate AI insights
      const response = await fetch('/api/llm/reports/generate-insights', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        
        // Update insights with real AI-generated data
        if (data.insights) {
          setAiInsights(data.insights);
        }
        
        // Update report sections to show completion
        setReportSections(prev => prev.map(section => ({
          ...section,
          progress: 100,
          status: 'completed',
          lastUpdated: 'Just now'
        })));
        
        console.log('AI Report generated successfully:', data);
      } else {
        console.error('Failed to generate AI report');
      }
    } catch (error) {
      console.error('Error generating AI report:', error);
    }
    
    setIsGenerating(false);
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
      <Card className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border-purple-200">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="w-8 h-8 text-purple-600" />
              <div>
                <CardTitle className="text-2xl">Llama AI Intelligence Report</CardTitle>
                <p className="text-muted-foreground">
                  Comprehensive analysis across 17 assets using advanced AI models
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm">
                <Eye className="w-4 h-4 mr-2" />
                View Full Report
              </Button>
              <Button onClick={generateAIReport} disabled={isGenerating}>
                {isGenerating ? (
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Zap className="w-4 h-4 mr-2" />
                )}
                {isGenerating ? 'Generating...' : 'Generate Report'}
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      <Tabs defaultValue="insights" className="space-y-6">
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
          {/* Generated Reports */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <PieChart className="w-5 h-5" />
                Generated Reports
              </CardTitle>
              <Button>
                <Download className="w-4 h-4 mr-2" />
                Export All
              </Button>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <Card className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold">Weekly AI Analysis</h3>
                    <Badge>Ready</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    Comprehensive 7-day analysis across all 17 assets with AI-powered insights and recommendations.
                  </p>
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline">
                      <Eye className="w-3 w-3 mr-1" />
                      Preview
                    </Button>
                    <Button size="sm">
                      <Download className="w-3 w-3 mr-1" />
                      Download
                    </Button>
                  </div>
                </Card>

                <Card className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold">Risk Assessment Report</h3>
                    <Badge variant="secondary">Processing</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    AI-powered risk analysis with portfolio optimization suggestions and stress testing results.
                  </p>
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline" disabled>
                      <RefreshCw className="w-3 w-3 mr-1 animate-spin" />
                      Processing
                    </Button>
                  </div>
                </Card>

                <Card className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold">Market Sentiment Analysis</h3>
                    <Badge>Ready</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    Social media and news sentiment analysis with predictive insights for market movements.
                  </p>
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline">
                      <Eye className="w-3 w-3 mr-1" />
                      Preview
                    </Button>
                    <Button size="sm">
                      <Download className="w-3 w-3 mr-1" />
                      Download
                    </Button>
                  </div>
                </Card>

                <Card className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold">Technical Analysis Report</h3>
                    <Badge>Ready</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    Advanced technical indicators and pattern recognition with AI-enhanced signal detection.
                  </p>
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline">
                      <Eye className="w-3 w-3 mr-1" />
                      Preview
                    </Button>
                    <Button size="sm">
                      <Download className="w-3 w-3 mr-1" />
                      Download
                    </Button>
                  </div>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 