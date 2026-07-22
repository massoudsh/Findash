import { Suspense } from 'react';
import { VisualizationContent } from '@/components/visualization/visualization-content';
import { ChartShowcase } from '@/components/ui/chart-showcase';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AnalysisAgentInsightsPanel } from '@/components/trading/analysis-agent-insights';

export default function VisualizationPage() {
  return (
    <div className="container mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">نمایش داده</h1>
        <p className="text-gray-400">
          نمودارهای تعاملی و تحلیل بصری برای داده‌های معاملاتی شما
        </p>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
        <div className="min-w-0">
      <Suspense fallback={<div className="text-center text-gray-400">در حال بارگذاری نمودارها...</div>}>
        <ChartShowcase />
        <VisualizationContent />
      </Suspense>
        </div>
        <aside className="hidden xl:block min-h-[360px]">
          <AnalysisAgentInsightsPanel />
        </aside>
      </div>

      {/* بخش تحلیل معاملات */}
      <Card className="bg-gray-900 border-gray-800">
        <CardHeader>
          <CardTitle className="text-white">تحلیل معاملات</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-gray-800 p-4 rounded">
              <h3 className="text-sm font-medium text-gray-300 mb-2">مجموع معاملات</h3>
              <div className="text-2xl font-bold text-white">۱,۲۴۷</div>
              <p className="text-xs text-green-500">۱۲%+ این ماه</p>
            </div>
            <div className="bg-gray-800 p-4 rounded">
              <h3 className="text-sm font-medium text-gray-300 mb-2">نرخ برد</h3>
              <div className="text-2xl font-bold text-white">۶۸.۳%</div>
              <p className="text-xs text-green-500">۲.۱%+ این ماه</p>
            </div>
            <div className="bg-gray-800 p-4 rounded">
              <h3 className="text-sm font-medium text-gray-300 mb-2">میانگین بازده</h3>
              <div className="text-2xl font-bold text-white">۲.۴%</div>
              <p className="text-xs text-red-500">۰.۳%- این ماه</p>
            </div>
            <div className="bg-gray-800 p-4 rounded">
              <h3 className="text-sm font-medium text-gray-300 mb-2">نسبت شارپ</h3>
              <div className="text-2xl font-bold text-white">۱.۸۵</div>
              <p className="text-xs text-green-500">۰.۱۲+ این ماه</p>
            </div>
          </div>

          <div className="mt-6 p-4 bg-gray-800 rounded">
            <h3 className="text-lg font-semibold text-white mb-4">نمای کلی عملکرد</h3>
            <div className="text-center text-gray-400">
              <div className="h-48 flex items-center justify-center border-2 border-dashed border-gray-600 rounded">
                داشبورد تحلیل معاملاتی بلادرنگ اینجا نمایش داده می‌شود
                <br />
                (اتصال به Grafana/TradingView)
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
