'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { toast } from '@/components/ui/toast';
import { Target, PlayCircle, TrendingUp, Sparkles } from 'lucide-react';
import { strategyTemplates } from './options-content';

export interface OptionsStrategyForTerminal {
  name: string;
  description?: string;
  category?: string;
}

interface OptionsStrategiesTabProps {
  onTradeInTerminal?: (strategy?: OptionsStrategyForTerminal) => void;
}

export function OptionsStrategiesTab({ onTradeInTerminal }: OptionsStrategiesTabProps) {
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [selectedStrategy, setSelectedStrategy] = useState<typeof strategyTemplates[0] | null>(null);
  const [builderOpen, setBuilderOpen] = useState(false);

  const categories = ['all', ...Array.from(new Set(strategyTemplates.map((s) => s.category)))];
  const filtered =
    categoryFilter === 'all'
      ? strategyTemplates
      : strategyTemplates.filter((s) => s.category === categoryFilter);

  const handleDeploy = (strategy: typeof strategyTemplates[0]) => {
    setSelectedStrategy(strategy);
    setBuilderOpen(true);
  };

  const handleTradeInTerminal = (strategy?: typeof strategyTemplates[0]) => {
    setBuilderOpen(false);
    const s = strategy ?? selectedStrategy;
    onTradeInTerminal?.(s ? { name: s.name, description: s.description, category: s.category } : undefined);
  };

  return (
    <div className="space-y-6 p-4 md:p-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Target className="h-5 w-5 text-emerald-400" />
            Options Strategies
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            Choose a strategy to deploy or trade in the terminal
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={categoryFilter} onValueChange={setCategoryFilter}>
            <SelectTrigger className="w-44">
              <SelectValue placeholder="Category" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              {categories.filter((c) => c !== 'all').map((cat) => (
                <SelectItem key={cat} value={cat}>{cat}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              const suggested = strategyTemplates[Math.floor(Math.random() * strategyTemplates.length)];
              setCategoryFilter(suggested.category);
              toast({
                title: 'AI Suggestion',
                description: `Try "${suggested.name}" — ${suggested.description ?? 'Popular strategy for current conditions.'}`,
                type: 'info',
              });
            }}
          >
            <Sparkles className="h-4 w-4 mr-1" />
            AI Suggest
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Strategy library</CardTitle>
          <CardDescription>
            Deploy advanced options strategies. Use &quot;Trade in Terminal&quot; to open the trading terminal.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {filtered.map((strategy) => (
              <Card
                key={strategy.name}
                className="bg-card/80 hover:border-primary/50 transition-colors cursor-pointer overflow-hidden"
              >
                <CardContent className="p-4 space-y-3">
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-2xl">{strategy.icon}</span>
                    <Badge
                      variant="secondary"
                      className="text-xs shrink-0"
                    >
                      {strategy.category}
                    </Badge>
                  </div>
                  <div>
                    <h3 className="font-semibold text-sm">{strategy.name}</h3>
                    <p className="text-xs text-muted-foreground line-clamp-2 mt-0.5">
                      {strategy.description}
                    </p>
                  </div>
                  <div className="flex gap-2 text-xs">
                    <Badge variant="outline" className="text-xs">
                      {strategy.complexity}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      {strategy.riskLevel}
                    </Badge>
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Popularity</span>
                      <span>{strategy.popularity}%</span>
                    </div>
                    <Progress value={strategy.popularity} className="h-1.5" />
                  </div>
                  <div className="flex gap-2 pt-1">
                    <Button
                      size="sm"
                      className="flex-1"
                      onClick={() => handleTradeInTerminal(strategy)}
                    >
                      <TrendingUp className="h-3 w-3 mr-1" />
                      Terminal
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="flex-1"
                      onClick={() => handleDeploy(strategy)}
                    >
                      <PlayCircle className="h-3 w-3 mr-1" />
                      Deploy
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      <Dialog open={builderOpen} onOpenChange={setBuilderOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {selectedStrategy?.icon && <span>{selectedStrategy.icon}</span>}
              {selectedStrategy?.name}
            </DialogTitle>
            <DialogDescription>
              {selectedStrategy?.description}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <p className="text-sm text-muted-foreground">
              Configure this strategy in the terminal with your chosen symbol and size.
            </p>
            <Button className="w-full" onClick={() => handleTradeInTerminal(selectedStrategy ?? undefined)}>
              <TrendingUp className="h-4 w-4 mr-2" />
              Open in Terminal
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
