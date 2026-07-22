'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Sparkles, ShieldAlert, Info } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getBackendUrl } from '@/lib/backend-url';
import { type AssetType } from './add-asset-modal';

// ─── Types (mirrors backend AllocationAnalysisResponse) ───────────────────────

interface CategoryBreakdown {
  type: AssetType;
  value: number;
  pct: number;
}

interface AllocationAnalysis {
  total_value: number;
  category_breakdown: CategoryBreakdown[];
  top_holding_pct: number;
  hhi: number;
  concentration_level: 'کم' | 'متوسط' | 'بالا';
  diversification_score: number;
  insights: string[];
  disclaimer: string;
}

export interface CopilotHolding {
  code: string;
  name: string;
  type: AssetType;
  value: number;
}

const LEVEL_STYLE: Record<AllocationAnalysis['concentration_level'], string> = {
  کم: 'bg-green-500/15 text-green-600 border-green-500/30',
  متوسط: 'bg-amber-500/15 text-amber-600 border-amber-500/30',
  بالا: 'bg-red-500/15 text-red-600 border-red-500/30',
};

export function AllocationCopilot({ holdings }: { holdings: CopilotHolding[] }) {
  const [analysis, setAnalysis] = useState<AllocationAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (holdings.length === 0) {
      setAnalysis(null);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetch(`${getBackendUrl()}/api/copilot/allocation-analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ holdings }),
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: AllocationAnalysis) => setAnalysis(data))
      .catch((err) => {
        if (err.name !== 'AbortError') setError('خطا در دریافت تحلیل تخصیص دارایی');
      })
      .finally(() => setLoading(false));

    return () => controller.abort();
  }, [holdings]);

  if (holdings.length === 0) return null;

  return (
    <Card dir="rtl" className="border-primary/20">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-1.5">
          <Sparkles className="h-4 w-4 text-primary" />
          دستیار هوشمند تخصیص دارایی
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {loading && (
          <p className="text-xs text-muted-foreground">در حال تحلیل ترکیب پرتفوی...</p>
        )}
        {error && <p className="text-xs text-red-500">{error}</p>}

        {analysis && (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className={cn('gap-1 text-xs', LEVEL_STYLE[analysis.concentration_level])}>
                <ShieldAlert className="h-3 w-3" />
                ریسک تمرکز: {analysis.concentration_level}
              </Badge>
              <Badge variant="outline" className="text-xs">
                امتیاز تنوع: {analysis.diversification_score}/۱۰۰
              </Badge>
              <Badge variant="outline" className="text-xs">
                بزرگ‌ترین دارایی: {analysis.top_holding_pct.toFixed(0)}٪
              </Badge>
            </div>

            {analysis.insights.length > 0 && (
              <ul className="space-y-1.5">
                {analysis.insights.map((insight, i) => (
                  <li key={i} className="text-xs text-muted-foreground flex items-start gap-1.5">
                    <span className="mt-1 h-1 w-1 rounded-full bg-primary flex-shrink-0" />
                    {insight}
                  </li>
                ))}
              </ul>
            )}

            <div className="flex items-start gap-1.5 rounded-lg bg-muted/40 p-2 text-[11px] text-muted-foreground">
              <Info className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
              {analysis.disclaimer}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
