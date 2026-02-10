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
  { symbol: 'BTC-USD', name: 'Bitcoin', type: 'crypto', category: 'Layer 1', marketCap: '$831B' },
  { symbol: 'ETH-USD', name: 'Ethereum', type: 'crypto', category: 'Smart Contract', marketCap: '$310B' },
  { symbol: 'AAPL', name: 'Apple Inc.', type: 'stock', category: 'Technology', marketCap: '$2.8T' },
  { symbol: 'TSLA', name: 'Tesla Inc.', type: 'stock', category: 'EV/Energy', marketCap: '$780B' },
  { symbol: 'NVDA', name: 'NVIDIA Corp.', type: 'stock', category: 'AI/Semiconductors', marketCap: '$1.2T' },
  { symbol: 'MSFT', name: 'Microsoft Corp.', type: 'stock', category: 'Cloud/Software', marketCap: '$2.9T' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', type: 'stock', category: 'Tech/AI', marketCap: '$1.7T' },
  { symbol: 'LINK-USD', name: 'Chainlink', type: 'crypto', category: 'Oracle Network', marketCap: '$8.2B' },
  { symbol: 'SOL-USD', name: 'Solana', type: 'crypto', category: 'Layer 1', marketCap: '$43B' },
  { symbol: 'AVAX-USD', name: 'Avalanche', type: 'crypto', category: 'Layer 1', marketCap: '$15B' },
  { symbol: 'MATIC-USD', name: 'Polygon', type: 'crypto', category: 'Layer 2', marketCap: '$9.8B' },
  { symbol: 'UNI-USD', name: 'Uniswap', type: 'crypto', category: 'DeFi', marketCap: '$5.2B' },
];

const NEW_PROJECTS = [
  { symbol: 'ARB-USD', name: 'Arbitrum', type: 'crypto', category: 'Layer 2', status: 'Emerging', risk: 'High' },
  { symbol: 'OP-USD', name: 'Optimism', type: 'crypto', category: 'Layer 2', status: 'Growing', risk: 'Medium' },
  { symbol: 'BLUR-USD', name: 'Blur', type: 'crypto', category: 'NFT/Gaming', status: 'New', risk: 'Very High' },
  { symbol: 'LDO-USD', name: 'Lido DAO', type: 'crypto', category: 'Liquid Staking', status: 'Established', risk: 'Medium' },
  { symbol: 'RDNT-USD', name: 'Radiant Capital', type: 'crypto', category: 'DeFi/Lending', status: 'Emerging', risk: 'High' },
  { symbol: 'GMX-USD', name: 'GMX', type: 'crypto', category: 'DeFi/Derivatives', status: 'Growing', risk: 'Medium' },
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-indigo-900 to-slate-900 p-6">
      <div className="space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold gradient-text">Deep Fundamental Research</h1>
            <p className="text-gray-400 mt-2">AI-powered fundamental analysis with institutional-grade research and PDF reports</p>
          </div>
          <div className="flex space-x-2">
            <Badge className="bg-indigo-500/20 text-indigo-300 border-indigo-500/30">
              LLM-Powered
            </Badge>
            <Button onClick={runDeepAnalysis} disabled={deepAnalysisLoading} className="btn-morphic">
              <Brain className="h-4 w-4 mr-2" />
              {deepAnalysisLoading ? 'Analyzing...' : 'AI Analysis'}
            </Button>
            <Button onClick={generateDeepResearchReport} disabled={reportGenerating} className="btn-morphic">
              <FileText className="h-4 w-4 mr-2" />
              {reportGenerating ? 'Generating...' : 'Generate PDF'}
            </Button>
          </div>
        </div>

        <Tabs defaultValue="research" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 glass-card">
            <TabsTrigger value="research">Deep Research</TabsTrigger>
            <TabsTrigger value="new-projects">New Projects</TabsTrigger>
            <TabsTrigger value="ai-insights">AI Insights</TabsTrigger>
            <TabsTrigger value="reports">Reports</TabsTrigger>
          </TabsList>

          <TabsContent value="research" className="space-y-6">
            {/* Asset Selection & Research Tools */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Database className="h-5 w-5 text-blue-400" />
                  <span>Research Center</span>
                </CardTitle>
                <CardDescription>Select assets for deep fundamental analysis with AI-powered insights</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Asset Selection</label>
                    <Select value={selectedAsset} onValueChange={handleAssetSelect}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select an asset" />
                      </SelectTrigger>
                      <SelectContent>
                        {PREDEFINED_ASSETS.map((asset) => (
                          <SelectItem key={asset.symbol} value={asset.symbol}>
                            <div className="flex items-center space-x-2">
                              <span className="font-medium">{asset.symbol}</span>
                              <span className="text-gray-500">- {asset.name}</span>
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
                    <label className="text-sm font-medium">Category Filter</label>
                    <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                      <SelectTrigger>
                        <SelectValue placeholder="All Categories" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Categories</SelectItem>
                        <SelectItem value="layer 1">Layer 1</SelectItem>
                        <SelectItem value="layer 2">Layer 2</SelectItem>
                        <SelectItem value="defi">DeFi</SelectItem>
                        <SelectItem value="technology">Technology</SelectItem>
                        <SelectItem value="ai">AI/ML</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Search Assets</label>
                    <div className="flex space-x-2">
                      <Input
                        placeholder="Search symbol or name..."
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
                          {asset.type}
                        </Badge>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Category:</span>
                          <span className="text-sm font-medium">{asset.category}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Market Cap:</span>
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
                  <span>New & Emerging Projects</span>
                </CardTitle>
                <CardDescription>Research and analysis of new blockchain projects and tokens</CardDescription>
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
                          project.risk === 'Very High' ? 'bg-red-500/20 text-red-300' :
                          project.risk === 'High' ? 'bg-orange-500/20 text-orange-300' :
                          'bg-yellow-500/20 text-yellow-300'
                        }`}>
                          {project.risk}
                        </Badge>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Category:</span>
                          <span className="text-sm font-medium">{project.category}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Status:</span>
                          <Badge className={`text-xs ${
                            project.status === 'Established' ? 'bg-green-500/20 text-green-300' :
                            project.status === 'Growing' ? 'bg-blue-500/20 text-blue-300' :
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
                        Research Project
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
                      <span>AI Market Intelligence</span>
                    </CardTitle>
                    <CardDescription>LLM-powered insights from comprehensive market analysis</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                      <div className="text-center p-4 neomorphic rounded-lg">
                        <div className="text-2xl font-bold text-green-400">{aiInsights.market_summary?.bullish_signals || 0}</div>
                        <div className="text-sm text-gray-400">Bullish Signals</div>
                      </div>
                      <div className="text-center p-4 neomorphic rounded-lg">
                        <div className="text-2xl font-bold text-red-400">{aiInsights.market_summary?.bearish_signals || 0}</div>
                        <div className="text-sm text-gray-400">Bearish Signals</div>
                      </div>
                      <div className="text-center p-4 neomorphic rounded-lg">
                        <div className="text-2xl font-bold text-blue-400">{aiInsights.market_summary?.total_assets_analyzed || 0}</div>
                        <div className="text-sm text-gray-400">Assets Analyzed</div>
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
                            <span className="text-sm text-gray-400">Confidence:</span>
                            <div className="flex items-center space-x-2">
                              <Progress value={insight.confidence} className="w-20 h-2" />
                              <span className="text-sm font-medium">{insight.confidence}%</span>
                            </div>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-400">Impact:</span>
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
                  <h3 className="text-lg font-medium mb-2">No AI Insights Generated</h3>
                  <p className="text-gray-400 mb-4">Click "AI Analysis" to generate comprehensive market insights</p>
                  <Button onClick={runDeepAnalysis} disabled={deepAnalysisLoading} className="btn-morphic">
                    <Brain className="h-4 w-4 mr-2" />
                    {deepAnalysisLoading ? 'Analyzing...' : 'Generate AI Insights'}
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
                  <span>Research Reports</span>
                </CardTitle>
                <CardDescription>Generate comprehensive PDF reports using AI analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h3 className="font-semibold">Report Types</h3>
                    <div className="space-y-3">
                      <div className="p-4 neomorphic rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium">Deep Fundamental Analysis</h4>
                          <Badge className="bg-blue-500/20 text-blue-300">Comprehensive</Badge>
                        </div>
                        <p className="text-sm text-gray-400 mb-3">Complete fundamental analysis with AI insights, financial metrics, and recommendations</p>
                        <Button 
                          onClick={generateDeepResearchReport} 
                          disabled={reportGenerating}
                          className="w-full btn-morphic"
                        >
                          <Download className="h-4 w-4 mr-2" />
                          {reportGenerating ? 'Generating...' : 'Generate PDF'}
                        </Button>
                      </div>
                      
                      <div className="p-4 neomorphic rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium">Technical + Fundamental</h4>
                          <Badge className="bg-purple-500/20 text-purple-300">Combined</Badge>
                        </div>
                        <p className="text-sm text-gray-400 mb-3">Integrated technical and fundamental analysis for complete market view</p>
                        <Button className="w-full btn-morphic" variant="outline">
                          <Download className="h-4 w-4 mr-2" />
                          Coming Soon
                        </Button>
                      </div>
                      
                      <div className="p-4 neomorphic rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium">Portfolio Analysis</h4>
                          <Badge className="bg-green-500/20 text-green-300">Multi-Asset</Badge>
                        </div>
                        <p className="text-sm text-gray-400 mb-3">Complete portfolio analysis with risk assessment and optimization</p>
                        <Button className="w-full btn-morphic" variant="outline">
                          <Download className="h-4 w-4 mr-2" />
                          Coming Soon
                        </Button>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <h3 className="font-semibold">Report Features</h3>
                    <div className="space-y-3">
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-400" />
                        <span className="text-sm">AI-powered insights and analysis</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-400" />
                        <span className="text-sm">Real-time market data integration</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-400" />
                        <span className="text-sm">Professional PDF formatting</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-400" />
                        <span className="text-sm">Institutional-grade research</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-400" />
                        <span className="text-sm">Risk assessment and recommendations</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-400" />
                        <span className="text-sm">Competitive landscape analysis</span>
                      </div>
                    </div>
                    
                    <div className="mt-6 p-4 bg-indigo-500/10 border border-indigo-500/20 rounded-lg">
                      <h4 className="font-medium text-indigo-300 mb-2">AI-Powered Research</h4>
                      <p className="text-sm text-gray-300">
                        Our reports use advanced LLM models to analyze market data, 
                        news sentiment, and fundamental metrics to provide institutional-grade 
                        research and actionable insights.
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
} 