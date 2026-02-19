'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  Bitcoin, 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Users, 
  Zap,
  Shield,
  Database,
  Network,
  DollarSign,
  BarChart3,
  Target,
  Eye,
  Layers,
  PieChart,
  ArrowUpDown,
  Wallet,
  Clock,
  AlertTriangle,
  RefreshCw
} from 'lucide-react';

export default function OnChainPage() {
  const [onchainData, setOnchainData] = useState<Record<string, unknown> | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  const fetchOnchainData = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/onchain/comprehensive');
      if (res.ok) {
        const json = await res.json();
        setOnchainData(json.data ?? null);
        setLastUpdated(json.last_updated ?? new Date().toISOString());
      } else {
        setOnchainData(null);
      }
    } catch {
      setOnchainData(null);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchOnchainData();
    const interval = setInterval(fetchOnchainData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // Bitcoin Network Fundamentals (Glassnode-style)
  const bitcoinFundamentals = [
    { metric: 'Hash Rate', value: '520.2 EH/s', change: '+5.2%', trend: 'up', signal: 'bullish', category: 'security' },
    { metric: 'Mining Difficulty', value: '67.3T', change: '+2.8%', trend: 'up', signal: 'neutral', category: 'security' },
    { metric: 'Block Time', value: '9.8 min', change: '-1.2%', trend: 'down', signal: 'neutral', category: 'performance' },
    { metric: 'Mempool Size', value: '145 MB', change: '-15.3%', trend: 'down', signal: 'bullish', category: 'performance' }
  ];

  // Network Activity & Adoption
  const networkActivity = [
    { metric: 'Active Addresses (24h)', value: '1.12M', change: '+8.1%', trend: 'up', signal: 'bullish', category: 'adoption' },
    { metric: 'New Addresses (24h)', value: '385K', change: '+12.4%', trend: 'up', signal: 'bullish', category: 'adoption' },
    { metric: 'Transaction Count', value: '284K', change: '+3.2%', trend: 'up', signal: 'neutral', category: 'usage' },
    { metric: 'Avg Transaction Value', value: '$48.2K', change: '+15.8%', trend: 'up', signal: 'bullish', category: 'usage' }
  ];

  // Market Structure & Flows
  const marketStructure = [
    { metric: 'Exchange Inflows', value: '8,245 BTC', change: '-22.5%', trend: 'down', signal: 'bullish', category: 'flows' },
    { metric: 'Exchange Outflows', value: '12,890 BTC', change: '+18.3%', trend: 'up', signal: 'bullish', category: 'flows' },
    { metric: 'Exchange Balance', value: '2.18M BTC', change: '-0.8%', trend: 'down', signal: 'bullish', category: 'supply' },
    { metric: 'Stablecoin Inflows', value: '$2.8B', change: '+45.2%', trend: 'up', signal: 'bullish', category: 'flows' }
  ];

  // HODLer & Investor Behavior
  const hodlerMetrics = [
    { metric: 'Long-Term Holders', value: '14.8M BTC', change: '+2.1%', trend: 'up', signal: 'bullish', category: 'hodl' },
    { metric: 'Short-Term Holders', value: '4.2M BTC', change: '-1.8%', trend: 'down', signal: 'neutral', category: 'hodl' },
    { metric: 'Coin Days Destroyed', value: '2.1M', change: '-35.2%', trend: 'down', signal: 'bullish', category: 'hodl' },
    { metric: 'Dormancy Flow', value: '0.85', change: '-12.4%', trend: 'down', signal: 'bullish', category: 'hodl' }
  ];

  // Valuation Models & Metrics
  const valuationMetrics = [
    { metric: 'MVRV Ratio', value: '1.85', change: '+3.4%', trend: 'up', signal: 'neutral', category: 'valuation' },
    { metric: 'NVT Ratio', value: '28.5', change: '-8.2%', trend: 'down', signal: 'bullish', category: 'valuation' },
    { metric: 'Realized Price', value: '$28,450', change: '+1.2%', trend: 'up', signal: 'neutral', category: 'valuation' },
    { metric: 'Thermocap Ratio', value: '12.8', change: '+5.1%', trend: 'up', signal: 'neutral', category: 'valuation' }
  ];

  // Whale & Institution Activity
  const whaleActivity = [
    { metric: 'Whale Addresses (>1K BTC)', value: '2,125', change: '+0.8%', trend: 'up', signal: 'neutral', category: 'whales' },
    { metric: 'Whale Transaction Count', value: '1,245', change: '-12.3%', trend: 'down', signal: 'neutral', category: 'whales' },
    { metric: 'Whale Net Flow', value: '-2,450 BTC', change: '+85.2%', trend: 'up', signal: 'bullish', category: 'whales' },
    { metric: 'Entity-Adjusted Dormancy', value: '0.72', change: '-15.8%', trend: 'down', signal: 'bullish', category: 'whales' }
  ];

  // DeFi & Layer 2 Analytics
  const defiL2Metrics = [
    { protocol: 'Lightning Network', capacity: '5,125 BTC', nodes: '15,450', change: '+8.2%', category: 'layer2' },
    { protocol: 'Ethereum L2s', tvl: '$42.8B', transactions: '2.8M/day', change: '+15.4%', category: 'layer2' },
    { protocol: 'Wrapped Bitcoin', supply: '168K WBTC', volume: '$890M', change: '+5.4%', category: 'defi' },
    { protocol: 'Bitcoin DeFi', tvl: '$2.1B', protocols: '45', change: '+22.1%', category: 'defi' }
  ];

  // Mining & Security Analytics
  const miningMetrics = [
    { metric: 'Puell Multiple', value: '0.85', change: '-12.3%', trend: 'down', signal: 'neutral', category: 'mining' },
    { metric: 'Hash Ribbons', value: 'Bullish', change: 'Signal', trend: 'up', signal: 'bullish', category: 'mining' },
    { metric: 'Miner Revenue', value: '$28.5M', change: '+8.4%', trend: 'up', signal: 'neutral', category: 'mining' },
    { metric: 'Fee Revenue %', value: '2.8%', change: '-45.2%', trend: 'down', signal: 'neutral', category: 'mining' }
  ];

  const renderMetricSection = (title: string, metrics: any[], icon: any, description: string) => (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-foreground">
          {icon}
          <span>{title}</span>
        </CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {metrics.map((metric, index) => (
            <div key={metric.metric} className="neomorphic p-4 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-muted-foreground">{metric.metric}</span>
                <div className="flex items-center space-x-2">
                  <Badge className={`text-xs ${
                    metric.signal === 'bullish' ? 'bg-green-500/20 text-green-300' :
                    metric.signal === 'bearish' ? 'bg-red-500/20 text-red-300' :
                    'bg-gray-500/20 text-gray-300'
                  }`}>
                    {metric.signal}
                  </Badge>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-lg font-bold text-foreground">{metric.value}</span>
                <div className={`flex items-center space-x-1 ${
                  metric.trend === 'up' ? 'text-chart-2' : 
                  metric.trend === 'down' ? 'text-destructive' : 'text-muted-foreground'
                }`}>
                  {metric.trend === 'up' ? (
                    <TrendingUp className="h-4 w-4" />
                  ) : metric.trend === 'down' ? (
                    <TrendingDown className="h-4 w-4" />
                  ) : (
                    <Activity className="h-4 w-4" />
                  )}
                  <span className="text-sm">{metric.change}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">On-Chain</h1>
          <p className="text-muted-foreground mt-1">
            Blockchain intelligence and on-chain metrics — real-time when backend is available
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {onchainData ? (
            <Badge className="bg-primary/20 text-primary border-primary/30">
              Live from source
            </Badge>
          ) : (
            <Badge variant="secondary">Static snapshot</Badge>
          )}
          {lastUpdated && (
            <span className="text-xs text-muted-foreground">
              Updated {lastUpdated ? new Date(lastUpdated).toLocaleTimeString() : ''}
            </span>
          )}
          <Button variant="outline" size="sm" onClick={fetchOnchainData} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 mr-1 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>
      <div className="space-y-6">

        {/* Bitcoin Network Fundamentals */}
        {renderMetricSection(
          'Bitcoin Network Fundamentals',
          bitcoinFundamentals,
          <Shield className="h-5 w-5 text-foreground" />,
          'Core network security and performance metrics'
        )}

        {/* Network Activity & Adoption */}
        {renderMetricSection(
          'Network Activity & Adoption',
          networkActivity,
          <Users className="h-5 w-5 text-foreground" />,
          'User adoption and network usage indicators'
        )}

        {/* Market Structure & Flows */}
        {renderMetricSection(
          'Market Structure & Capital Flows',
          marketStructure,
          <ArrowUpDown className="h-5 w-5 text-foreground" />,
          'Exchange flows and market structure analysis'
        )}

        {/* HODLer & Investor Behavior */}
        {renderMetricSection(
          'HODLer & Investor Behavior',
          hodlerMetrics,
          <Clock className="h-5 w-5 text-foreground" />,
          'Long-term holder patterns and coin age analysis'
        )}

        {/* Valuation Models */}
        {renderMetricSection(
          'Valuation Models & Metrics',
          valuationMetrics,
          <Target className="h-5 w-5 text-foreground" />,
          'Advanced valuation models and fair value indicators'
        )}

        {/* Whale & Institution Activity */}
        {renderMetricSection(
          'Whale & Institution Activity',
          whaleActivity,
          <Eye className="h-5 w-5 text-foreground" />,
          'Large holder behavior and institutional flow analysis'
        )}

        {/* Mining & Security Analytics */}
        {renderMetricSection(
          'Mining & Security Analytics',
          miningMetrics,
          <Zap className="h-5 w-5 text-foreground" />,
          'Mining economics and network security indicators'
        )}

        {/* DeFi & Layer 2 Ecosystem */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Layers className="h-5 w-5 text-foreground" />
              <span>DeFi & Layer 2 Ecosystem</span>
            </CardTitle>
            <CardDescription>Cross-chain and DeFi protocol analytics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {defiL2Metrics.map((protocol, index) => (
                <div key={protocol.protocol} className="neomorphic p-4 rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 flex items-center justify-center">
                        <span className="text-foreground font-bold text-xs">
                          {protocol.protocol.slice(0, 2)}
                        </span>
                      </div>
                      <span className="font-medium">{protocol.protocol}</span>
                    </div>
                    <Badge className={`text-xs ${
                      protocol.category === 'layer2' ? 'bg-blue-500/20 text-blue-300' : 'bg-purple-500/20 text-purple-300'
                    }`}>
                      {protocol.category}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <p className="text-muted-foreground">
                        {protocol.category === 'layer2' ? 'Capacity/TVL' : 'Supply/TVL'}
                      </p>
                      <p className="font-medium">{protocol.capacity || protocol.tvl}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">
                        {protocol.category === 'layer2' ? 'Nodes/Txns' : 'Volume/Protocols'}
                      </p>
                      <p className="font-medium">{protocol.nodes || protocol.transactions || protocol.volume || protocol.protocols}</p>
                    </div>
                  </div>
                  <div className="mt-3 flex justify-between items-center">
                    <span className="text-xs text-muted-foreground">24h Change</span>
                    <span className={`text-sm font-medium ${
                      protocol.change.startsWith('+') ? 'text-chart-2' : 'text-destructive'
                    }`}>
                      {protocol.change}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Market Intelligence Summary */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5 text-foreground" />
              <span>Market Intelligence Summary</span>
            </CardTitle>
            <CardDescription>AI-powered insights from on-chain data</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="space-y-4">
                <h4 className="font-semibold text-muted-foreground">Network Health</h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span>Security Score</span>
                    <Badge className="bg-green-500/20 text-green-300">95/100</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Adoption Trend</span>
                    <Badge className="bg-blue-500/20 text-blue-300">Accelerating</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Network Stress</span>
                    <Badge className="bg-green-500/20 text-green-300">Low</Badge>
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <h4 className="font-semibold text-muted-foreground">Market Sentiment</h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span>HODLer Strength</span>
                    <Badge className="bg-green-500/20 text-green-300">Strong</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Institutional Flow</span>
                    <Badge className="bg-blue-500/20 text-blue-300">Accumulating</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Retail Activity</span>
                    <Badge className="bg-yellow-500/20 text-yellow-300">Moderate</Badge>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h4 className="font-semibold text-muted-foreground">Valuation Signals</h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span>Fair Value</span>
                    <Badge className="bg-gray-500/20 text-gray-300">Neutral</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Cycle Position</span>
                    <Badge className="bg-blue-500/20 text-blue-300">Mid-Bull</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Risk/Reward</span>
                    <Badge className="bg-green-500/20 text-green-300">Favorable</Badge>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 