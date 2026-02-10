'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  Download, 
  FileText, 
  Table, 
  File,
  Settings,
  Calendar,
  Filter,
  CheckCircle,
  AlertCircle
} from 'lucide-react';

interface ExportOption {
  id: string;
  name: string;
  description: string;
  format: 'csv' | 'xlsx' | 'pdf' | 'json';
  icon: any;
  fileSize?: string;
}

interface DataExportProps {
  data?: any[];
  filename?: string;
  title?: string;
  className?: string;
}

export function DataExport({ 
  data = [], 
  filename = 'export', 
  title = 'Export Data',
  className 
}: DataExportProps) {
  const [isExporting, setIsExporting] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState<string>('csv');
  const [exportStatus, setExportStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const exportOptions: ExportOption[] = [
    {
      id: 'csv',
      name: 'CSV',
      description: 'Comma-separated values file',
      format: 'csv',
      icon: Table,
      fileSize: '~2KB'
    },
    {
      id: 'xlsx',
      name: 'Excel',
      description: 'Microsoft Excel spreadsheet',
      format: 'xlsx',
      icon: FileText,
      fileSize: '~5KB'
    },
    {
      id: 'pdf',
      name: 'PDF',
      description: 'Portable document format',
      format: 'pdf',
      icon: File,
      fileSize: '~10KB'
    },
    {
      id: 'json',
      name: 'JSON',
      description: 'JavaScript object notation',
      format: 'json',
      icon: Settings,
      fileSize: '~3KB'
    }
  ];

  // Mock data if none provided
  const mockData = [
    { symbol: 'AAPL', price: 150.25, change: 2.15, volume: 89543210, date: '2024-01-20' },
    { symbol: 'TSLA', price: 202.50, change: -5.75, volume: 45234567, date: '2024-01-20' },
    { symbol: 'NVDA', price: 875.30, change: 15.80, volume: 32154876, date: '2024-01-20' },
    { symbol: 'MSFT', price: 378.90, change: 3.45, volume: 28453691, date: '2024-01-20' },
  ];

  const exportData = data.length > 0 ? data : mockData;

  const generateCSV = (data: any[]) => {
    if (data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const csvContent = [
      headers.join(','),
      ...data.map(row => 
        headers.map(header => {
          const value = row[header];
          // Escape commas and quotes
          if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
            return `"${value.replace(/"/g, '""')}"`;
          }
          return value;
        }).join(',')
      )
    ].join('\n');
    
    return csvContent;
  };

  const generateJSON = (data: any[]) => {
    return JSON.stringify(data, null, 2);
  };

  const downloadFile = (content: string, filename: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleExport = async (format: string) => {
    setIsExporting(true);
    setExportStatus('idle');

    try {
      // Simulate processing delay
      await new Promise(resolve => setTimeout(resolve, 1500));

      const timestamp = new Date().toISOString().split('T')[0];
      
      switch (format) {
        case 'csv':
          const csvContent = generateCSV(exportData);
          downloadFile(csvContent, `${filename}-${timestamp}.csv`, 'text/csv');
          break;
          
        case 'json':
          const jsonContent = generateJSON(exportData);
          downloadFile(jsonContent, `${filename}-${timestamp}.json`, 'application/json');
          break;
          
        case 'xlsx':
          // For Excel, we'd use a library like SheetJS in a real implementation
          const csvForExcel = generateCSV(exportData);
          downloadFile(csvForExcel, `${filename}-${timestamp}.xlsx`, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
          break;
          
        case 'pdf':
          // For PDF, we'd use a library like jsPDF in a real implementation
          const pdfContent = `# ${title}\n\n${generateCSV(exportData)}`;
          downloadFile(pdfContent, `${filename}-${timestamp}.pdf`, 'application/pdf');
          break;
          
        default:
          throw new Error('Unsupported format');
      }
      
      setExportStatus('success');
    } catch (error) {
      console.error('Export failed:', error);
      setExportStatus('error');
    } finally {
      setIsExporting(false);
      
      // Reset status after 3 seconds
      setTimeout(() => {
        setExportStatus('idle');
      }, 3000);
    }
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Download className="h-5 w-5" />
          {title}
          {exportStatus === 'success' && (
            <Badge variant="default" className="bg-green-100 text-green-800">
              <CheckCircle className="h-3 w-3 mr-1" />
              Exported
            </Badge>
          )}
          {exportStatus === 'error' && (
            <Badge variant="destructive">
              <AlertCircle className="h-3 w-3 mr-1" />
              Failed
            </Badge>
          )}
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Export your trading data in various formats
        </p>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Export Options */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {exportOptions.map((option) => {
            const Icon = option.icon;
            return (
              <button
                key={option.id}
                onClick={() => setSelectedFormat(option.id)}
                disabled={isExporting}
                className={`p-4 border-2 rounded-lg transition-all hover:border-primary/50 ${
                  selectedFormat === option.id 
                    ? 'border-primary bg-primary/5' 
                    : 'border-gray-200 dark:border-gray-700'
                } ${isExporting ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
              >
                <div className="flex flex-col items-center text-center space-y-2">
                  <Icon className="h-8 w-8 text-primary" />
                  <div>
                    <div className="font-medium text-sm">{option.name}</div>
                    <div className="text-xs text-muted-foreground">{option.description}</div>
                    <div className="text-xs text-muted-foreground mt-1">{option.fileSize}</div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Export Settings */}
        <div className="space-y-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h4 className="font-medium text-sm flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Export Settings
          </h4>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Date Range:</span>
              <span className="font-medium">Last 30 days</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Records:</span>
              <span className="font-medium">{exportData.length} items</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Format:</span>
              <span className="font-medium">{exportOptions.find(o => o.id === selectedFormat)?.name}</span>
            </div>
          </div>
        </div>

        {/* Preview */}
        <div className="space-y-2">
          <h4 className="font-medium text-sm">Data Preview</h4>
          <div className="border rounded-lg overflow-hidden">
            <div className="bg-gray-50 dark:bg-gray-800 px-3 py-2 border-b">
              <div className="grid grid-cols-5 gap-2 text-xs font-medium text-muted-foreground">
                <span>Symbol</span>
                <span>Price</span>
                <span>Change</span>
                <span>Volume</span>
                <span>Date</span>
              </div>
            </div>
            <div className="max-h-32 overflow-y-auto">
              {exportData.slice(0, 3).map((row, index) => (
                <div key={index} className="grid grid-cols-5 gap-2 px-3 py-2 text-xs border-b last:border-b-0">
                  <span className="font-medium">{row.symbol}</span>
                  <span>${row.price}</span>
                  <span className={row.change >= 0 ? 'text-green-600' : 'text-red-600'}>
                    {row.change >= 0 ? '+' : ''}{row.change}
                  </span>
                  <span>{row.volume?.toLocaleString()}</span>
                  <span>{row.date}</span>
                </div>
              ))}
              {exportData.length > 3 && (
                <div className="px-3 py-2 text-xs text-muted-foreground text-center">
                  +{exportData.length - 3} more rows...
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Export Button */}
        <Button 
          onClick={() => handleExport(selectedFormat)}
          disabled={isExporting}
          className="w-full"
          size="lg"
        >
          {isExporting ? (
            <>
              <Download className="h-4 w-4 mr-2 animate-pulse" />
              Exporting...
            </>
          ) : (
            <>
              <Download className="h-4 w-4 mr-2" />
              Export as {exportOptions.find(o => o.id === selectedFormat)?.name}
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
} 