import { Suspense } from 'react';
import { VisualizationContent } from '@/components/visualization/visualization-content';
import { ChartShowcase } from '@/components/ui/chart-showcase';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function VisualizationPage() {
  return (
    <div className="container mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">Visualization</h1>
        <p className="text-gray-400">
          Interactive charts and visual analytics for your trading data
        </p>
      </div>
      
      <Suspense fallback={<div className="text-center text-gray-400">Loading visualizations...</div>}>
        <ChartShowcase />
        <VisualizationContent />
      </Suspense>

      {/* Trading Analytics Section */}
      <Card className="bg-gray-900 border-gray-800">
        <CardHeader>
          <CardTitle className="text-white">Trading Analytics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-gray-800 p-4 rounded">
              <h3 className="text-sm font-medium text-gray-300 mb-2">Total Trades</h3>
              <div className="text-2xl font-bold text-white">1,247</div>
              <p className="text-xs text-green-500">+12% this month</p>
            </div>
            <div className="bg-gray-800 p-4 rounded">
              <h3 className="text-sm font-medium text-gray-300 mb-2">Win Rate</h3>
              <div className="text-2xl font-bold text-white">68.3%</div>
              <p className="text-xs text-green-500">+2.1% this month</p>
            </div>
            <div className="bg-gray-800 p-4 rounded">
              <h3 className="text-sm font-medium text-gray-300 mb-2">Avg. Return</h3>
              <div className="text-2xl font-bold text-white">2.4%</div>
              <p className="text-xs text-red-500">-0.3% this month</p>
            </div>
            <div className="bg-gray-800 p-4 rounded">
              <h3 className="text-sm font-medium text-gray-300 mb-2">Sharpe Ratio</h3>
              <div className="text-2xl font-bold text-white">1.85</div>
              <p className="text-xs text-green-500">+0.12 this month</p>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-gray-800 rounded">
            <h3 className="text-lg font-semibold text-white mb-4">Performance Overview</h3>
            <div className="text-center text-gray-400">
              <div className="h-48 flex items-center justify-center border-2 border-dashed border-gray-600 rounded">
                Real-time trading analytics dashboard would be embedded here
                <br />
                (Grafana/TradingView integration)
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 