'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  TrendingUp, 
  TrendingDown,
  BarChart3, 
  DollarSign,
  Activity,
  Brain,
  Shield,
  Target,
  Bell,
  RefreshCw,
  Settings,
  Plus,
  Minus,
  Maximize2,
  Minimize2,
  X,
  GripVertical,
  Eye,
  EyeOff,
  Edit,
  Save,
  RotateCcw,
  Download,
  Star,
  Zap,
  Clock,
  Users,
  PieChart,
  LineChart,
  ArrowUp,
  ArrowDown,
  Calendar,
  Globe,
  Database,
  Cpu,
  HardDrive,
  Wifi
} from 'lucide-react';

interface Widget {
  id: string;
  title: string;
  type: 'metric' | 'chart' | 'list' | 'activity' | 'progress' | 'alert';
  size: 'small' | 'medium' | 'large';
  position: { x: number; y: number };
  visible: boolean;
  refreshable: boolean;
  configurable: boolean;
  icon: any;
  data?: any;
  config?: any;
}

interface DashboardLayout {
  id: string;
  name: string;
  widgets: Widget[];
  columns: number;
}

const LAYOUTS_STORAGE_KEY = 'octopus_dashboard_layouts';
const CURRENT_LAYOUT_KEY = 'octopus_current_layout';

// Default widgets
const defaultWidgets: Widget[] = [
  {
    id: 'portfolio-value',
    title: 'Portfolio Value',
    type: 'metric',
    size: 'medium',
    position: { x: 0, y: 0 },
    visible: true,
    refreshable: true,
    configurable: false,
    icon: DollarSign,
    data: { value: 125847.32, change: 2.47, changePercent: 2.01 }
  },
  {
    id: 'daily-pnl',
    title: 'Daily P&L',
    type: 'metric',
    size: 'medium',
    position: { x: 1, y: 0 },
    visible: true,
    refreshable: true,
    configurable: false,
    icon: TrendingUp,
    data: { value: 2847.32, change: 147.83, changePercent: 5.47 }
  },
  {
    id: 'active-positions',
    title: 'Active Positions',
    type: 'metric',
    size: 'medium',
    position: { x: 2, y: 0 },
    visible: true,
    refreshable: true,
    configurable: false,
    icon: Target,
    data: { value: 12, change: 2, changePercent: 20 }
  },
  {
    id: 'market-status',
    title: 'Market Status',
    type: 'alert',
    size: 'small',
    position: { x: 3, y: 0 },
    visible: true,
    refreshable: true,
    configurable: false,
    icon: Activity,
    data: { status: 'open', message: 'Markets are open', type: 'success' }
  },
  {
    id: 'performance-chart',
    title: 'Performance Overview',
    type: 'chart',
    size: 'large',
    position: { x: 0, y: 1 },
    visible: true,
    refreshable: true,
    configurable: true,
    icon: BarChart3,
    data: { 
      chartData: [
        { date: '2024-01', value: 120000 },
        { date: '2024-02', value: 122000 },
        { date: '2024-03', value: 125000 },
        { date: '2024-04', value: 128000 },
        { date: '2024-05', value: 125847 }
      ]
    }
  },
  {
    id: 'top-gainers',
    title: 'Top Gainers',
    type: 'list',
    size: 'medium',
    position: { x: 2, y: 1 },
    visible: true,
    refreshable: true,
    configurable: true,
    icon: TrendingUp,
    data: {
      items: [
        { symbol: 'AAPL', change: 5.47, price: 185.92 },
        { symbol: 'TSLA', change: 3.82, price: 248.41 },
        { symbol: 'NVDA', change: 2.15, price: 875.28 },
        { symbol: 'MSFT', change: 1.23, price: 378.85 }
      ]
    }
  },
  {
    id: 'ai-insights',
    title: 'AI Insights',
    type: 'activity',
    size: 'medium',
    position: { x: 0, y: 2 },
    visible: true,
    refreshable: true,
    configurable: true,
    icon: Brain,
    data: {
      insights: [
        { text: 'Strong buy signal detected for AAPL', confidence: 85, time: '2 min ago' },
        { text: 'Tech sector showing bullish momentum', confidence: 72, time: '5 min ago' },
        { text: 'Portfolio optimization suggested', confidence: 68, time: '1 hour ago' }
      ]
    }
  },
  {
    id: 'risk-metrics',
    title: 'Risk Metrics',
    type: 'progress',
    size: 'medium',
    position: { x: 1, y: 2 },
    visible: true,
    refreshable: true,
    configurable: false,
    icon: Shield,
    data: {
      metrics: [
        { name: 'VaR (95%)', value: 2.4, max: 5, status: 'good' },
        { name: 'Beta', value: 1.15, max: 2, status: 'warning' },
        { name: 'Sharpe Ratio', value: 1.84, max: 3, status: 'good' }
      ]
    }
  },
  {
    id: 'system-status',
    title: 'System Health',
    type: 'metric',
    size: 'small',
    position: { x: 2, y: 2 },
    visible: true,
    refreshable: true,
    configurable: false,
    icon: Cpu,
    data: { value: 98.5, change: 0.2, unit: '%', status: 'excellent' }
  }
];

const defaultLayouts: DashboardLayout[] = [
  {
    id: 'default',
    name: 'Default Layout',
    widgets: defaultWidgets,
    columns: 4
  },
  {
    id: 'trader',
    name: 'Trader Focus',
    widgets: defaultWidgets.filter(w => 
      ['portfolio-value', 'daily-pnl', 'active-positions', 'performance-chart', 'top-gainers'].includes(w.id)
    ),
    columns: 3
  },
  {
    id: 'analyst',
    name: 'Analyst View',
    widgets: defaultWidgets.filter(w => 
      ['performance-chart', 'ai-insights', 'risk-metrics', 'system-status'].includes(w.id)
    ),
    columns: 2
  }
];

export function WidgetSystem() {
  const [layouts, setLayouts] = useState<DashboardLayout[]>(defaultLayouts);
  const [currentLayoutId, setCurrentLayoutId] = useState('default');
  const [editMode, setEditMode] = useState(false);
  const [draggedWidget, setDraggedWidget] = useState<string | null>(null);
  const [hoveredPosition, setHoveredPosition] = useState<{ x: number; y: number } | null>(null);

  const currentLayout = layouts.find(l => l.id === currentLayoutId) || layouts[0];

  // Load saved layouts and current layout
  useEffect(() => {
    const savedLayouts = localStorage.getItem(LAYOUTS_STORAGE_KEY);
    const savedCurrentLayout = localStorage.getItem(CURRENT_LAYOUT_KEY);
    
    if (savedLayouts) {
      try {
        setLayouts(JSON.parse(savedLayouts));
      } catch (e) {
        console.error('Failed to load layouts:', e);
      }
    }
    
    if (savedCurrentLayout) {
      setCurrentLayoutId(savedCurrentLayout);
    }
  }, []);

  // Save layouts and current layout
  const saveLayouts = useCallback((newLayouts: DashboardLayout[]) => {
    setLayouts(newLayouts);
    localStorage.setItem(LAYOUTS_STORAGE_KEY, JSON.stringify(newLayouts));
  }, []);

  const saveCurrentLayout = useCallback((layoutId: string) => {
    setCurrentLayoutId(layoutId);
    localStorage.setItem(CURRENT_LAYOUT_KEY, layoutId);
  }, []);

  // Handle drag start
  const handleDragStart = (e: React.DragEvent, widgetId: string) => {
    setDraggedWidget(widgetId);
    e.dataTransfer.effectAllowed = 'move';
  };

  // Handle drag over
  const handleDragOver = (e: React.DragEvent, x: number, y: number) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setHoveredPosition({ x, y });
  };

  // Handle drop
  const handleDrop = (e: React.DragEvent, x: number, y: number) => {
    e.preventDefault();
    
    if (!draggedWidget) return;

    const newLayouts = layouts.map(layout => {
      if (layout.id === currentLayoutId) {
        const newWidgets = layout.widgets.map(widget => {
          if (widget.id === draggedWidget) {
            return { ...widget, position: { x, y } };
          }
          return widget;
        });
        return { ...layout, widgets: newWidgets };
      }
      return layout;
    });

    saveLayouts(newLayouts);
    setDraggedWidget(null);
    setHoveredPosition(null);
  };

  // Toggle widget visibility
  const toggleWidgetVisibility = (widgetId: string) => {
    const newLayouts = layouts.map(layout => {
      if (layout.id === currentLayoutId) {
        const newWidgets = layout.widgets.map(widget => {
          if (widget.id === widgetId) {
            return { ...widget, visible: !widget.visible };
          }
          return widget;
        });
        return { ...layout, widgets: newWidgets };
      }
      return layout;
    });
    saveLayouts(newLayouts);
  };

  // Reset to default layout
  const resetLayout = () => {
    const newLayouts = layouts.map(layout => {
      if (layout.id === currentLayoutId) {
        const defaultLayout = defaultLayouts.find(l => l.id === currentLayoutId);
        return defaultLayout || layout;
      }
      return layout;
    });
    saveLayouts(newLayouts);
  };

  // Refresh widget data
  const refreshWidget = (widgetId: string) => {
    console.log(`Refreshing widget: ${widgetId}`);
    // In a real app, this would fetch fresh data
  };

  // Render widget content based on type
  const renderWidgetContent = (widget: Widget) => {
    const Icon = widget.icon;

    switch (widget.type) {
      case 'metric':
        return (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Icon className="h-5 w-5 text-muted-foreground" />
              {widget.data?.status && (
                <Badge 
                  className={
                    widget.data.status === 'excellent' ? 'bg-green-500/20 text-green-300' :
                    widget.data.status === 'good' ? 'bg-blue-500/20 text-blue-300' :
                    'bg-yellow-500/20 text-yellow-300'
                  }
                >
                  {widget.data.status}
                </Badge>
              )}
            </div>
            <div>
              <div className="text-2xl font-bold">
                {widget.data?.value?.toLocaleString() || '0'}
                {widget.data?.unit && <span className="text-sm ml-1">{widget.data.unit}</span>}
              </div>
              {widget.data?.change !== undefined && (
                <div className={`flex items-center text-sm ${
                  widget.data.change >= 0 ? 'text-green-500' : 'text-red-500'
                }`}>
                  {widget.data.change >= 0 ? 
                    <ArrowUp className="h-3 w-3 mr-1" /> : 
                    <ArrowDown className="h-3 w-3 mr-1" />
                  }
                  {widget.data.change >= 0 ? '+' : ''}
                  {widget.data.change.toLocaleString()}
                  {widget.data.changePercent && (
                    <span className="ml-1">
                      ({widget.data.changePercent >= 0 ? '+' : ''}
                      {widget.data.changePercent.toFixed(2)}%)
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        );

      case 'list':
        return (
          <div className="space-y-3">
            <div className="flex items-center">
              <Icon className="h-5 w-5 text-muted-foreground mr-2" />
              <span className="text-sm text-muted-foreground">
                {widget.data?.items?.length || 0} items
              </span>
            </div>
            <div className="space-y-2">
              {(widget.data?.items || []).map((item: any, index: number) => (
                <div key={index} className="flex items-center justify-between p-2 rounded bg-muted">
                  <span className="font-medium">{item.symbol}</span>
                  <div className="text-right">
                    <div className="font-bold">${item.price}</div>
                    <div className={`text-xs ${item.change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );

      case 'activity':
        return (
          <div className="space-y-3">
            <div className="flex items-center">
              <Icon className="h-5 w-5 text-muted-foreground mr-2" />
              <span className="text-sm text-muted-foreground">Recent insights</span>
            </div>
            <div className="space-y-3">
              {(widget.data?.insights || []).map((insight: any, index: number) => (
                <div key={index} className="space-y-2">
                  <div className="text-sm">{insight.text}</div>
                  <div className="flex items-center justify-between">
                    <Progress value={insight.confidence} className="flex-1" />
                    <span className="text-xs text-muted-foreground ml-2">
                      {insight.confidence}%
                    </span>
                  </div>
                  <div className="text-xs text-muted-foreground">{insight.time}</div>
                </div>
              ))}
            </div>
          </div>
        );

      case 'progress':
        return (
          <div className="space-y-3">
            <div className="flex items-center">
              <Icon className="h-5 w-5 text-muted-foreground mr-2" />
              <span className="text-sm text-muted-foreground">Risk levels</span>
            </div>
            <div className="space-y-3">
              {(widget.data?.metrics || []).map((metric: any, index: number) => (
                <div key={index} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{metric.name}</span>
                    <span className="text-sm">{metric.value}</span>
                  </div>
                  <Progress 
                    value={(metric.value / metric.max) * 100} 
                    className={`${
                      metric.status === 'good' ? 'text-green-500' :
                      metric.status === 'warning' ? 'text-yellow-500' :
                      'text-red-500'
                    }`}
                  />
                </div>
              ))}
            </div>
          </div>
        );

      case 'alert':
        return (
          <div className="space-y-3">
            <div className="flex items-center">
              <Icon className="h-5 w-5 text-muted-foreground mr-2" />
              <Badge 
                className={
                  widget.data?.type === 'success' ? 'bg-green-500/20 text-green-300' :
                  widget.data?.type === 'warning' ? 'bg-yellow-500/20 text-yellow-300' :
                  'bg-red-500/20 text-red-300'
                }
              >
                {widget.data?.status || 'Unknown'}
              </Badge>
            </div>
            <div className="text-sm">{widget.data?.message || 'No message'}</div>
          </div>
        );

      case 'chart':
        return (
          <div className="space-y-3">
            <div className="flex items-center">
              <Icon className="h-5 w-5 text-muted-foreground mr-2" />
              <span className="text-sm text-muted-foreground">
                {widget.data?.chartData?.length || 0} data points
              </span>
            </div>
            <div className="h-32 flex items-center justify-center border-2 border-dashed border-muted rounded">
              <div className="text-center text-muted-foreground">
                <BarChart3 className="h-8 w-8 mx-auto mb-2" />
                <div className="text-sm">Chart visualization would be here</div>
              </div>
            </div>
          </div>
        );

      default:
        return (
          <div className="text-center text-muted-foreground">
            <Icon className="h-8 w-8 mx-auto mb-2" />
            <div>Widget content</div>
          </div>
        );
    }
  };

  // Create grid positions
  const gridPositions = [];
  const maxRows = Math.max(...currentLayout.widgets.map(w => w.position.y)) + 2;
  
  for (let y = 0; y < maxRows; y++) {
    for (let x = 0; x < currentLayout.columns; x++) {
      gridPositions.push({ x, y });
    }
  }

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h2 className="text-2xl font-bold">Dashboard</h2>
          <select 
            value={currentLayoutId}
            onChange={(e) => saveCurrentLayout(e.target.value)}
            className="px-3 py-1 rounded border bg-background"
          >
            {layouts.map(layout => (
              <option key={layout.id} value={layout.id}>
                {layout.name}
              </option>
            ))}
          </select>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant={editMode ? "default" : "outline"}
            size="sm"
            onClick={() => setEditMode(!editMode)}
          >
            <Edit className="h-4 w-4 mr-2" />
            {editMode ? 'Done' : 'Edit'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={resetLayout}
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => window.location.reload()}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh All
          </Button>
        </div>
      </div>

      {/* Edit Mode Instructions */}
      {editMode && (
        <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
          <div className="flex items-center space-x-2 text-blue-300">
            <Edit className="h-4 w-4" />
            <span className="font-medium">Edit Mode Active</span>
          </div>
          <div className="text-sm text-blue-300/80 mt-1">
            Drag widgets to reposition them. Use the eye icon to show/hide widgets.
          </div>
        </div>
      )}

      {/* Grid Layout */}
      <div 
        className="relative"
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${currentLayout.columns}, 1fr)`,
          gap: '1rem',
          minHeight: '600px'
        }}
      >
        {/* Grid Drop Zones (only visible in edit mode) */}
        {editMode && gridPositions.map(pos => {
          const isOccupied = currentLayout.widgets.some(w => 
            w.visible && w.position.x === pos.x && w.position.y === pos.y
          );
          const isHovered = hoveredPosition?.x === pos.x && hoveredPosition?.y === pos.y;
          
          if (isOccupied && !isHovered) return null;
          
          return (
            <div
              key={`${pos.x}-${pos.y}`}
              className={`min-h-[200px] border-2 border-dashed rounded-lg flex items-center justify-center transition-colors ${
                isHovered 
                  ? 'border-primary bg-primary/10' 
                  : 'border-muted-foreground/20 hover:border-muted-foreground/40'
              }`}
              style={{
                gridColumn: pos.x + 1,
                gridRow: pos.y + 1
              }}
              onDragOver={(e) => handleDragOver(e, pos.x, pos.y)}
              onDrop={(e) => handleDrop(e, pos.x, pos.y)}
            >
              {isHovered && draggedWidget && (
                <div className="text-primary font-medium">Drop here</div>
              )}
            </div>
          );
        })}

        {/* Widgets */}
        {currentLayout.widgets
          .filter(widget => widget.visible)
          .map(widget => {
            const Icon = widget.icon;
            
            return (
              <Card
                key={widget.id}
                className={`transition-all duration-200 ${
                  editMode ? 'cursor-move hover:shadow-lg border-2' : ''
                } ${
                  draggedWidget === widget.id ? 'opacity-50' : ''
                } ${
                  widget.size === 'small' ? 'min-h-[150px]' :
                  widget.size === 'medium' ? 'min-h-[200px]' :
                  'min-h-[300px]'
                } ${
                  widget.size === 'large' ? 'col-span-2' : ''
                }`}
                style={{
                  gridColumn: widget.position.x + 1,
                  gridRow: widget.position.y + 1
                }}
                draggable={editMode}
                onDragStart={(e) => handleDragStart(e, widget.id)}
              >
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base flex items-center space-x-2">
                      {editMode && <GripVertical className="h-4 w-4 text-muted-foreground" />}
                      <span>{widget.title}</span>
                    </CardTitle>
                    <div className="flex items-center space-x-1">
                      {editMode && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleWidgetVisibility(widget.id)}
                        >
                          {widget.visible ? 
                            <Eye className="h-3 w-3" /> : 
                            <EyeOff className="h-3 w-3" />
                          }
                        </Button>
                      )}
                      {widget.refreshable && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => refreshWidget(widget.id)}
                        >
                          <RefreshCw className="h-3 w-3" />
                        </Button>
                      )}
                      {widget.configurable && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => console.log(`Configure ${widget.id}`)}
                        >
                          <Settings className="h-3 w-3" />
                        </Button>
                      )}
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {renderWidgetContent(widget)}
                </CardContent>
              </Card>
            );
          })}
      </div>

      {/* Hidden Widgets Panel (in edit mode) */}
      {editMode && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Hidden Widgets</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {currentLayout.widgets
                .filter(widget => !widget.visible)
                .map(widget => {
                  const Icon = widget.icon;
                  return (
                    <div
                      key={widget.id}
                      className="p-3 border rounded-lg flex items-center space-x-2 opacity-50 hover:opacity-100 transition-opacity"
                    >
                      <Icon className="h-4 w-4" />
                      <span className="text-sm">{widget.title}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => toggleWidgetVisibility(widget.id)}
                      >
                        <Eye className="h-3 w-3" />
                      </Button>
                    </div>
                  );
                })}
            </div>
            {currentLayout.widgets.filter(w => !w.visible).length === 0 && (
              <div className="text-center text-muted-foreground py-8">
                All widgets are currently visible
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
} 