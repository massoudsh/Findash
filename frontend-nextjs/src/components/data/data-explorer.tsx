'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Database, Search, RefreshCw, Download } from 'lucide-react';

const SAMPLE_DATA = [
  { symbol: 'AAPL', date: '2024-01-15', open: 182.3, high: 184.9, low: 181.1, close: 183.5, volume: 52_340_000 },
  { symbol: 'TSLA', date: '2024-01-15', open: 245.6, high: 251.2, low: 243.0, close: 248.4, volume: 88_120_000 },
  { symbol: 'MSFT', date: '2024-01-15', open: 371.0, high: 376.5, low: 369.8, close: 374.2, volume: 20_780_000 },
  { symbol: 'GOOGL', date: '2024-01-15', open: 139.4, high: 141.8, low: 138.9, close: 140.7, volume: 18_950_000 },
  { symbol: 'NVDA', date: '2024-01-15', open: 495.2, high: 512.4, low: 492.0, close: 508.6, volume: 43_670_000 },
  { symbol: 'BTC-USD', date: '2024-01-15', open: 42_800, high: 43_600, low: 42_400, close: 43_200, volume: 28_900_000_000 },
];

export function DataExplorer() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);

  const filtered = SAMPLE_DATA.filter(
    (r) =>
      !query ||
      r.symbol.toLowerCase().includes(query.toLowerCase()) ||
      r.date.includes(query)
  );

  const handleRefresh = () => {
    setLoading(true);
    setTimeout(() => setLoading(false), 800);
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2 text-base">
          <Database className="h-4 w-4" />
          Data Explorer
          <Badge variant="secondary" className="text-xs">{filtered.length} rows</Badge>
        </CardTitle>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleRefresh} disabled={loading}>
            <RefreshCw className={`h-3 w-3 mr-1 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button variant="outline" size="sm">
            <Download className="h-3 w-3 mr-1" />
            Export
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="relative mb-4">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Filter by symbol or date..."
            className="pl-8 h-9"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div className="overflow-x-auto rounded-lg border">
          <table className="w-full text-sm">
            <thead className="bg-muted/50">
              <tr>
                {['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'].map((h) => (
                  <th key={h} className="text-left px-3 py-2 text-xs font-semibold text-muted-foreground">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((row, i) => (
                <tr key={i} className="border-t hover:bg-muted/20 transition-colors">
                  <td className="px-3 py-2 font-semibold">{row.symbol}</td>
                  <td className="px-3 py-2 text-muted-foreground">{row.date}</td>
                  <td className="px-3 py-2 font-mono">${row.open.toLocaleString()}</td>
                  <td className="px-3 py-2 font-mono text-green-600">${row.high.toLocaleString()}</td>
                  <td className="px-3 py-2 font-mono text-red-500">${row.low.toLocaleString()}</td>
                  <td className="px-3 py-2 font-mono font-semibold">${row.close.toLocaleString()}</td>
                  <td className="px-3 py-2 text-muted-foreground">{row.volume.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
