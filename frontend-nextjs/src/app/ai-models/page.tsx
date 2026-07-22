"use client";

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { MLModelsAgentPanel } from '@/components/agents/ml-models-agent-panel';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { 
  Brain, 
  Zap, 
  TrendingUp, 
  Activity, 
  PlayCircle, 
  StopCircle,
  RefreshCw,
  Eye,
  Download,
  Upload,
  Settings,
  Cpu,
  Database,
  BarChart3,
  Sparkles,
  Bot,
  Target,
  Layers,
  Gauge,
  LineChart,
  GitBranch,
  Workflow,
  Infinity,
  Network,
  Binary
} from 'lucide-react';

interface TrainingJob {
  id: string;
  model_type: string;
  symbol: string;
  status: 'running' | 'completed' | 'failed' | 'pending';
  progress: number;
  epochs_completed: number;
  total_epochs: number;
  loss: number;
  accuracy: number;
  started_at: string;
  estimated_completion: string;
  task_id?: string;
}

interface ModelMetrics {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'training' | 'idle';
  accuracy: number;
  last_trained: string;
  predictions_count: number;
  performance_score: number;
  description: string;
  complexity: 'low' | 'medium' | 'high' | 'extreme';
}

interface GenerativeConfig {
  input_dim: number;
  hidden_dim: number;
  latent_dim: number;
  learning_rate: number;
  num_epochs: number;
  batch_size: number;
}

interface XGBoostConfig {
  n_estimators: number;
  learning_rate: number;
  max_depth: number;
  objective: string;
  early_stopping_rounds: number;
}

export default function AIModelsPage() {
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [models, setModels] = useState<ModelMetrics[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [selectedModelTypes, setSelectedModelTypes] = useState(['transformer']);
  const [epochs, setEpochs] = useState(50);
  const [isLoading, setIsLoading] = useState(false);
  const [llmStatus, setLlmStatus] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('overview');
  
  // XGBoost Configuration
  const [xgboostConfig, setXgboostConfig] = useState<XGBoostConfig>({
    n_estimators: 500,
    learning_rate: 0.05,
    max_depth: 5,
    objective: 'reg:squarederror',
    early_stopping_rounds: 50
  });
  
  // GAN Configuration
  const [ganConfig, setGanConfig] = useState<GenerativeConfig>({
    input_dim: 60,
    hidden_dim: 128,
    latent_dim: 32,
    learning_rate: 0.001,
    num_epochs: 50,
    batch_size: 64
  });

  // Mock data initialization with new models
  useEffect(() => {
    const mockJobs: TrainingJob[] = [
      {
        id: '1',
        model_type: 'transformer',
        symbol: 'AAPL',
        status: 'running',
        progress: 75,
        epochs_completed: 38,
        total_epochs: 50,
        loss: 0.0023,
        accuracy: 0.94,
        started_at: '2024-01-15T10:30:00Z',
        estimated_completion: '2024-01-15T12:45:00Z'
      },
      {
        id: '2',
        model_type: 'xgboost',
        symbol: 'BTC-USD',
        status: 'completed',
        progress: 100,
        epochs_completed: 500,
        total_epochs: 500,
        loss: 0.0015,
        accuracy: 0.97,
        started_at: '2024-01-15T08:00:00Z',
        estimated_completion: '2024-01-15T10:30:00Z'
      },
      {
        id: '3',
        model_type: 'gan',
        symbol: 'ETH-USD',
        status: 'running',
        progress: 60,
        epochs_completed: 30,
        total_epochs: 50,
        loss: 0.0045,
        accuracy: 0.88,
        started_at: '2024-01-15T11:00:00Z',
        estimated_completion: '2024-01-15T13:30:00Z'
      }
    ];

    const mockModels: ModelMetrics[] = [
      {
        id: '1',
        name: 'AAPL-Transformer-v2.1',
        type: 'transformer',
        status: 'active',
        accuracy: 0.94,
        last_trained: '2024-01-15T10:30:00Z',
        predictions_count: 15420,
        performance_score: 0.89,
        description: 'مدل پیشرفته مبتنی بر توجه برای تشخیص الگوهای متوالی',
        complexity: 'high'
      },
      {
        id: '2',
        name: 'BTC-XGBoost-v3.2',
        type: 'xgboost',
        status: 'active',
        accuracy: 0.97,
        last_trained: '2024-01-15T08:00:00Z',
        predictions_count: 12847,
        performance_score: 0.95,
        description: 'مدل تجمیعی گرادیان بوستینگ برای پیش‌بینی دقیق قیمت',
        complexity: 'medium'
      },
      {
        id: '3',
        name: 'Portfolio-LLama-v3.0',
        type: 'llm',
        status: 'training',
        accuracy: 0.88,
        last_trained: '2024-01-15T06:00:00Z',
        predictions_count: 2847,
        performance_score: 0.85,
        description: 'مدل زبانی بزرگ برای تحلیل احساسات بازار',
        complexity: 'extreme'
      },
      {
        id: '4',
        name: 'ETH-GAN-Synthetic-v1.5',
        type: 'gan',
        status: 'active',
        accuracy: 0.91,
        last_trained: '2024-01-15T12:00:00Z',
        predictions_count: 8932,
        performance_score: 0.87,
        description: 'شبکه مولد تخاصمی برای تولید داده مصنوعی',
        complexity: 'high'
      },
      {
        id: '5',
        name: 'Multi-Asset-Prophet-v2.8',
        type: 'prophet',
        status: 'active',
        accuracy: 0.86,
        last_trained: '2024-01-14T16:00:00Z',
        predictions_count: 9654,
        performance_score: 0.82,
        description: 'پیش‌بینی سری زمانی با تجزیه فصلی',
        complexity: 'low'
      },
      {
        id: '6',
        name: 'Options-AutoEncoder-v1.3',
        type: 'autoencoder',
        status: 'idle',
        accuracy: 0.83,
        last_trained: '2024-01-14T14:00:00Z',
        predictions_count: 5421,
        performance_score: 0.79,
        description: 'مدل تشخیص ناهنجاری و فشرده‌سازی ویژگی',
        complexity: 'medium'
      }
    ];

    setTrainingJobs(mockJobs);
    setModels(mockModels);
    fetchLLMStatus();
  }, []);

  const fetchLLMStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/models/system-status');
      const data = await response.json();
      setLlmStatus(data);
    } catch (error) {
      console.error('Failed to fetch LLM status:', error);
    }
  };

  const startTraining = async () => {
    setIsLoading(true);
    try {
      let endpoint = '/api/training';
      let payload: any = {
        symbol: selectedSymbol,
        model_types: selectedModelTypes,
        epochs
      };

      // Use specific endpoints for different model types
      if (selectedModelTypes.includes('xgboost')) {
        endpoint = 'http://localhost:8000/api/ml-models/train';
        payload = {
          model_type: 'xgboost',
          symbol: selectedSymbol,
          start_date: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          end_date: new Date().toISOString().split('T')[0],
          hyperparameters: xgboostConfig
        };
      } else if (selectedModelTypes.includes('gan')) {
        endpoint = 'http://localhost:8000/api/generative/train';
        payload = {
          symbol: selectedSymbol,
          config: ganConfig
        };
      } else {
        endpoint = 'http://localhost:8000/api/training';
      }
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      const result = await response.json();
      console.log('Training started:', result);
      
      // Add new training job to list
      const newJob: TrainingJob = {
        id: result.task_id || result.job_id || Date.now().toString(),
        model_type: selectedModelTypes[0],
        symbol: selectedSymbol,
        status: 'pending',
        progress: 0,
        epochs_completed: 0,
        total_epochs: epochs,
        loss: 0,
        accuracy: 0,
        started_at: new Date().toISOString(),
        estimated_completion: new Date(Date.now() + epochs * 2 * 60 * 1000).toISOString(),
        task_id: result.task_id || result.job_id
      };
      
      setTrainingJobs(prev => [newJob, ...prev]);
    } catch (error) {
      console.error('Failed to start training:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const startLLMFineTuning = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/llm/finetune', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          output_dir: './peft-output'
        })
      });
      
      const result = await response.json();
      console.log('LLM Fine-tuning started:', result);
    } catch (error) {
      console.error('Failed to start LLM fine-tuning:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const generateAIInsights = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/llm/reports/generate-insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbols: [selectedSymbol],
          analysis_type: 'comprehensive'
        })
      });
      
      const result = await response.json();
      console.log('AI Insights generated:', result);
    } catch (error) {
      console.error('Failed to generate insights:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const generateSyntheticData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/generative/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_samples: 1000,
          symbol: selectedSymbol,
          config: ganConfig
        })
      });
      
      const result = await response.json();
      console.log('Synthetic data generated:', result);
    } catch (error) {
      console.error('Failed to generate synthetic data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': case 'training': case 'active': return 'bg-green-500';
      case 'completed': return 'bg-blue-500';
      case 'failed': return 'bg-red-500';
      case 'pending': case 'idle': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'low': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'high': return 'text-orange-400';
      case 'extreme': return 'text-red-400';
      default: return 'text-muted-foreground';
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'running': return 'در حال اجرا';
      case 'training': return 'در حال آموزش';
      case 'active': return 'فعال';
      case 'completed': return 'تکمیل‌شده';
      case 'failed': return 'ناموفق';
      case 'pending': return 'در انتظار';
      case 'idle': return 'بی‌کار';
      default: return status;
    }
  };

  const getComplexityLabel = (complexity: string) => {
    switch (complexity) {
      case 'low': return 'کم';
      case 'medium': return 'متوسط';
      case 'high': return 'بالا';
      case 'extreme': return 'بسیار بالا';
      default: return complexity;
    }
  };

  const getModelTypeLabel = (type: string) => {
    switch (type) {
      case 'transformer': return 'ترنسفورمر';
      case 'xgboost': return 'ایکس‌جی‌بوست';
      case 'gan': return 'گن (GAN)';
      case 'llm': return 'مدل زبانی';
      case 'prophet': return 'پروفت';
      case 'autoencoder': return 'اتوانکودر';
      default: return type;
    }
  };

  const getModelIcon = (type: string) => {
    switch (type) {
      case 'transformer': return <Brain className="h-5 w-5" />;
      case 'xgboost': return <Target className="h-5 w-5" />;
      case 'gan': return <Infinity className="h-5 w-5" />;
      case 'llm': return <Bot className="h-5 w-5" />;
      case 'prophet': return <TrendingUp className="h-5 w-5" />;
      case 'autoencoder': return <Network className="h-5 w-5" />;
      default: return <Cpu className="h-5 w-5" />;
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">مدل‌های هوش مصنوعی</h1>
        <p className="text-muted-foreground mt-1">
          آموزش و مدیریت مدل‌های یادگیری ماشین برای هوشمندی معاملاتی
        </p>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
        <div className="min-w-0 space-y-6">

      {/* Stats */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4"
      >
        <Card className="glass-card p-4 border-border/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10">
              <Cpu className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">مدل‌های فعال</p>
              <p className="text-xl font-bold text-foreground">{models.filter(m => m.status === 'active').length}</p>
            </div>
          </div>
        </Card>
        <Card className="glass-card p-4 border-border/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-chart-1/20">
              <Activity className="h-5 w-5 text-chart-1" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">وظایف آموزشی</p>
              <p className="text-xl font-bold text-foreground">{trainingJobs.filter(j => j.status === 'running').length}</p>
            </div>
          </div>
        </Card>
        <Card className="glass-card p-4 border-border/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-chart-2/20">
              <BarChart3 className="h-5 w-5 text-chart-2" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">میانگین دقت</p>
              <p className="text-xl font-bold text-foreground">
                {(models.reduce((acc, m) => acc + m.accuracy, 0) / models.length * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </Card>
        <Card className="glass-card p-4 border-border/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-chart-3/20">
              <Sparkles className="h-5 w-5 text-chart-3" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">پیش‌بینی‌های امروز</p>
              <p className="text-xl font-bold text-foreground">
                {models.reduce((acc, m) => acc + m.predictions_count, 0).toLocaleString()}
              </p>
            </div>
          </div>
        </Card>
        <Card className="glass-card p-4 border-border/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-chart-4/20">
              <Binary className="h-5 w-5 text-chart-4" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">انواع مدل</p>
              <p className="text-xl font-bold text-foreground">{new Set(models.map(m => m.type)).size}</p>
            </div>
          </div>
        </Card>
      </motion.div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <div className="border-b border-border/50 bg-card/30 px-4 py-2 rounded-lg">
          <TabsList className="grid w-full max-w-2xl grid-cols-4">
            <TabsTrigger value="overview" className="flex items-center gap-2">
              <Gauge className="h-4 w-4" />
              نمای کلی
            </TabsTrigger>
            <TabsTrigger value="training" className="flex items-center gap-2">
              <Zap className="h-4 w-4" />
              آموزش
            </TabsTrigger>
            <TabsTrigger value="models" className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              مدل‌ها
            </TabsTrigger>
            <TabsTrigger value="advanced" className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              پیشرفته
            </TabsTrigger>
          </TabsList>
        </div>

        {/* Overview Tab */}
        <TabsContent value="overview" className="mt-6 space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Model Performance Overview */}
            <Card className="glass-card p-6">
              <h2 className="text-2xl font-bold tracking-tight text-foreground mb-6 flex items-center gap-2">
                <Gauge className="h-6 w-6 text-foreground" />
                نمای کلی عملکرد مدل‌ها
              </h2>
              
              <div className="space-y-4">
                {models.slice(0, 4).map((model) => (
                  <div key={model.id} className="border border-border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getModelIcon(model.type)}
                        <h3 className="font-semibold text-foreground">{model.name}</h3>
                      </div>
                      <Badge className={`${getStatusColor(model.status)} text-foreground`}>
                        {getStatusLabel(model.status)}
                      </Badge>
                    </div>

                    <p className="text-sm text-muted-foreground mb-3">{model.description}</p>

                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">دقت</p>
                        <p className="text-green-400 font-bold">{(model.accuracy * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">عملکرد</p>
                        <p className="text-blue-400 font-bold">{(model.performance_score * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">پیچیدگی</p>
                        <p className={`font-bold ${getComplexityColor(model.complexity)}`}>
                          {getComplexityLabel(model.complexity)}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            {/* Quick Actions */}
            <Card className="glass-card p-6">
              <h2 className="text-2xl font-bold tracking-tight text-foreground mb-6 flex items-center gap-2">
                <Zap className="h-6 w-6 text-foreground" />
                اقدامات سریع
              </h2>

              <div className="space-y-4">
                <div>
                  <Label>نماد معاملاتی</Label>
                  <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
                    <SelectTrigger className="mt-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="AAPL">AAPL</SelectItem>
                      <SelectItem value="TSLA">TSLA</SelectItem>
                      <SelectItem value="BTC-USD">BTC-USD</SelectItem>
                      <SelectItem value="ETH-USD">ETH-USD</SelectItem>
                      <SelectItem value="NVDA">NVDA</SelectItem>
                      <SelectItem value="MSFT">MSFT</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <Button 
                    onClick={() => setActiveTab('training')}
                  >
                    <PlayCircle className="h-4 w-4 mr-2" />
                    شروع آموزش
                  </Button>

                  <Button
                    onClick={generateAIInsights}
                    disabled={isLoading}
                    variant="outline"
                  >
                    <Sparkles className="h-4 w-4 mr-2" />
                    بینش‌های هوش مصنوعی
                  </Button>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <Button
                    onClick={generateSyntheticData}
                    disabled={isLoading}
                    variant="secondary"
                  >
                    <Infinity className="h-4 w-4 mr-2" />
                    تولید داده
                  </Button>

                  <Button
                    onClick={startLLMFineTuning}
                    disabled={isLoading}
                    variant="outline"
                  >
                    <Bot className="h-4 w-4 mr-2" />
                    فاین‌تیون مدل زبانی
                  </Button>
                </div>
              </div>
            </Card>
          </div>
        </TabsContent>

        {/* Training Tab */}
        <TabsContent value="training" className="mt-6 space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Training Controls */}
            <Card className="glass-card p-6">
              <h2 className="text-2xl font-bold tracking-tight text-foreground mb-6 flex items-center gap-2">
                <Zap className="h-6 w-6 text-foreground" />
                شروع آموزش جدید
              </h2>

              <div className="space-y-4">
                <div>
                  <Label>نماد معاملاتی</Label>
                  <Input
                    value={selectedSymbol}
                    onChange={(e) => setSelectedSymbol(e.target.value)}
                    placeholder="مثلا AAPL, TSLA, BTC-USD"
                    className="mt-1"
                  />
                </div>

                <div>
                  <Label>انواع مدل</Label>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {['transformer', 'xgboost', 'gan', 'prophet', 'autoencoder'].map((type) => (
                      <Button
                        key={type}
                        variant={selectedModelTypes.includes(type) ? "default" : "outline"}
                        size="sm"
                        onClick={() => {
                          setSelectedModelTypes(prev => 
                            prev.includes(type) 
                              ? prev.filter(t => t !== type)
                              : [...prev, type]
                          );
                        }}
                        className="flex items-center gap-1"
                      >
                        {getModelIcon(type)}
                        {getModelTypeLabel(type)}
                      </Button>
                    ))}
                  </div>
                </div>

                <div>
                  <Label>تعداد epoch آموزش</Label>
                  <Input
                    type="number"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value))}
                    min="10"
                    max="500"
                    className="mt-1"
                  />
                </div>

                <Button 
                  onClick={startTraining}
                  disabled={isLoading || selectedModelTypes.length === 0}
                  className="w-full"
                >
                  <PlayCircle className="h-4 w-4 mr-2" />
                  شروع آموزش
                </Button>
              </div>
            </Card>

            {/* XGBoost Configuration */}
            <Card className="glass-card p-6">
              <h2 className="text-2xl font-bold tracking-tight text-foreground mb-6 flex items-center gap-2">
                <Target className="h-6 w-6 text-foreground" />
                تنظیمات XGBoost
              </h2>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>تعداد Estimator</Label>
                    <Input
                      type="number"
                      value={xgboostConfig.n_estimators}
                      onChange={(e) => setXgboostConfig(prev => ({
                        ...prev,
                        n_estimators: parseInt(e.target.value)
                      }))}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label>نرخ یادگیری</Label>
                    <Input
                      type="number"
                      step="0.01"
                      value={xgboostConfig.learning_rate}
                      onChange={(e) => setXgboostConfig(prev => ({
                        ...prev,
                        learning_rate: parseFloat(e.target.value)
                      }))}
                      className="mt-1"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>حداکثر عمق</Label>
                    <Input
                      type="number"
                      value={xgboostConfig.max_depth}
                      onChange={(e) => setXgboostConfig(prev => ({
                        ...prev,
                        max_depth: parseInt(e.target.value)
                      }))}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label>توقف زودهنگام</Label>
                    <Input
                      type="number"
                      value={xgboostConfig.early_stopping_rounds}
                      onChange={(e) => setXgboostConfig(prev => ({
                        ...prev,
                        early_stopping_rounds: parseInt(e.target.value)
                      }))}
                      className="mt-1"
                    />
                  </div>
                </div>

                <div>
                  <Label>هدف مدل</Label>
                  <Select
                    value={xgboostConfig.objective}
                    onValueChange={(value) => setXgboostConfig(prev => ({
                      ...prev,
                      objective: value
                    }))}
                  >
                    <SelectTrigger className="mt-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="reg:squarederror">رگرسیون (خطای مربعی)</SelectItem>
                      <SelectItem value="reg:absoluteerror">رگرسیون (خطای مطلق)</SelectItem>
                      <SelectItem value="binary:logistic">طبقه‌بندی دودویی</SelectItem>
                      <SelectItem value="multi:softmax">طبقه‌بندی چندکلاسه</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </Card>
          </div>

          {/* GAN Configuration */}
          <Card className="glass-card p-6">
            <h2 className="text-2xl font-bold tracking-tight text-foreground mb-6 flex items-center gap-2">
              <Infinity className="h-6 w-6 text-foreground" />
              تنظیمات GAN
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-muted-foreground">معماری</h3>
                <div>
                  <Label>بعد ورودی</Label>
                  <Input
                    type="number"
                    value={ganConfig.input_dim}
                    onChange={(e) => setGanConfig(prev => ({
                      ...prev,
                      input_dim: parseInt(e.target.value)
                    }))}
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label>بعد پنهان</Label>
                  <Input
                    type="number"
                    value={ganConfig.hidden_dim}
                    onChange={(e) => setGanConfig(prev => ({
                      ...prev,
                      hidden_dim: parseInt(e.target.value)
                    }))}
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label>بعد نهفته (Latent)</Label>
                  <Input
                    type="number"
                    value={ganConfig.latent_dim}
                    onChange={(e) => setGanConfig(prev => ({
                      ...prev,
                      latent_dim: parseInt(e.target.value)
                    }))}
                    className="mt-1"
                  />
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-muted-foreground">آموزش</h3>
                <div>
                  <Label>نرخ یادگیری</Label>
                  <Input
                    type="number"
                    step="0.0001"
                    value={ganConfig.learning_rate}
                    onChange={(e) => setGanConfig(prev => ({
                      ...prev,
                      learning_rate: parseFloat(e.target.value)
                    }))}
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label>تعداد Epoch</Label>
                  <Input
                    type="number"
                    value={ganConfig.num_epochs}
                    onChange={(e) => setGanConfig(prev => ({
                      ...prev,
                      num_epochs: parseInt(e.target.value)
                    }))}
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label>اندازه دسته (Batch)</Label>
                  <Input
                    type="number"
                    value={ganConfig.batch_size}
                    onChange={(e) => setGanConfig(prev => ({
                      ...prev,
                      batch_size: parseInt(e.target.value)
                    }))}
                    className="mt-1"
                  />
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-muted-foreground">اقدامات</h3>
                <div className="space-y-2">
                  <Button
                    onClick={generateSyntheticData}
                    disabled={isLoading}
                    className="w-full"
                  >
                    <Infinity className="h-4 w-4 mr-2" />
                    تولید داده مصنوعی
                  </Button>

                  <Button
                    variant="outline"
                    className="w-full"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    خروجی گرفتن از مدل GAN
                  </Button>

                  <Button
                    variant="outline"
                    className="w-full"
                  >
                    <Eye className="h-4 w-4 mr-2" />
                    نمایش فضای نهفته
                  </Button>
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>

        {/* Models Tab */}
        <TabsContent value="models" className="mt-6 space-y-6">
          {/* Active Models */}
          <Card className="glass-card p-6">
            <h2 className="text-2xl font-bold tracking-tight text-foreground mb-6 flex items-center gap-2">
              <Database className="h-6 w-6 text-foreground" />
              فهرست مدل‌های فعال
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {models.map((model) => (
                <div key={model.id} className="border border-border rounded-lg p-4 hover:border-primary/30 transition-colors">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getModelIcon(model.type)}
                      <h3 className="font-semibold text-foreground">{model.name}</h3>
                    </div>
                    <Badge className={`${getStatusColor(model.status)} text-foreground`}>
                      {getStatusLabel(model.status)}
                    </Badge>
                  </div>

                  <p className="text-sm text-muted-foreground mb-3">{model.description}</p>

                  <div className="grid grid-cols-2 gap-4 text-sm mb-3">
                    <div>
                      <p className="text-muted-foreground">دقت</p>
                      <p className="text-green-400 font-bold">{(model.accuracy * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">عملکرد</p>
                      <p className="text-blue-400 font-bold">{(model.performance_score * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">پیش‌بینی‌ها</p>
                      <p className="text-purple-400 font-bold">{model.predictions_count.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">پیچیدگی</p>
                      <p className={`font-bold ${getComplexityColor(model.complexity)}`}>
                        {getComplexityLabel(model.complexity)}
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button size="sm" variant="outline" className="flex-1">
                      <Eye className="h-3 w-3 mr-1" />
                      نمایش
                    </Button>
                    <Button size="sm" variant="outline" className="flex-1">
                      <Download className="h-3 w-3 mr-1" />
                      خروجی
                    </Button>
                    <Button size="sm" variant="outline" className="flex-1">
                      <Settings className="h-3 w-3 mr-1" />
                      تنظیمات
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </TabsContent>

        {/* Advanced Tab */}
        <TabsContent value="advanced" className="mt-6 space-y-6">
          {/* Training Jobs */}
          <Card className="glass-card p-6">
            <h2 className="text-2xl font-bold tracking-tight text-foreground mb-6 flex items-center gap-2">
              <TrendingUp className="h-6 w-6 text-foreground" />
              پایش وظایف آموزشی
            </h2>

            <div className="space-y-4">
              {trainingJobs.map((job) => (
                <div key={job.id} className="border border-border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      {getModelIcon(job.model_type)}
                      <h3 className="font-semibold text-foreground">
                        {job.symbol} - {getModelTypeLabel(job.model_type)}
                      </h3>
                      <Badge className={`${getStatusColor(job.status)} text-foreground`}>
                        {getStatusLabel(job.status)}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button size="sm" variant="outline">
                        <RefreshCw className="h-3 w-3" />
                      </Button>
                      <Button size="sm" variant="outline">
                        <StopCircle className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>

                  {job.status === 'running' && (
                    <div className="mb-3">
                      <div className="flex justify-between text-sm mb-1">
                        <span>پیشرفت: {job.epochs_completed}/{job.total_epochs} epoch</span>
                        <span>{job.progress}%</span>
                      </div>
                      <Progress value={job.progress} className="h-2" />
                    </div>
                  )}

                  <div className="grid grid-cols-4 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">خطا (Loss)</p>
                      <p className="text-red-400 font-bold">{job.loss.toFixed(4)}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">دقت</p>
                      <p className="text-green-400 font-bold">{(job.accuracy * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">شروع</p>
                      <p className="text-blue-400">{new Date(job.started_at).toLocaleTimeString()}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">زمان تخمینی پایان</p>
                      <p className="text-purple-400">{new Date(job.estimated_completion).toLocaleTimeString()}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* System Status */}
          <Card className="glass-card p-6">
            <h2 className="text-2xl font-bold tracking-tight text-foreground mb-6 flex items-center gap-2">
              <Activity className="h-6 w-6 text-foreground" />
              وضعیت سیستم
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-muted-foreground">سخت‌افزار</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">GPU در دسترس:</span>
                    <span className="text-green-400">✓ CUDA 11.8</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">مصرف حافظه:</span>
                    <span className="text-yellow-400">۶۷٪</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">هسته‌های CPU:</span>
                    <span className="text-blue-400">۱۶ هسته</span>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-muted-foreground">نسخه مدل‌ها</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">PyTorch:</span>
                    <span className="text-green-400">2.1.0</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">XGBoost:</span>
                    <span className="text-green-400">1.7.3</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Transformers:</span>
                    <span className="text-green-400">4.36.0</span>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-muted-foreground">عملکرد</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">میانگین زمان آموزش:</span>
                    <span className="text-blue-400" dir="ltr">۲.۳ ساعت</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">سرعت استنتاج:</span>
                    <span className="text-green-400" dir="ltr">۴۵ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">دقت مدل:</span>
                    <span className="text-purple-400">۹۱.۲٪</span>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
        </div>
        <aside className="hidden xl:block xl:sticky xl:top-6 h-[calc(100vh-14rem)] min-h-[420px] max-h-[560px] overflow-y-auto">
          <MLModelsAgentPanel />
        </aside>
      </div>
    </div>
  );
} 