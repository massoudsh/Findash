import { DataExplorer } from "@/components/data/data-explorer";
import { DataExport } from "@/components/ui/data-export";
import { DataCollectorAgentPanel } from '@/components/trading/data-collector-agent-panel';

export default function DataExplorerPage() {
  return (
    <div className="container mx-auto py-6">
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
        <div className="min-w-0">
          <h1 className="text-3xl font-bold mb-6">Data Explorer</h1>
          <div className="space-y-6">
            <DataExport
              title="Export Trading Data"
              filename="trading-data"
              className="mb-6"
            />
            <DataExplorer />
          </div>
        </div>
        <aside className="hidden xl:block min-h-[360px]">
          <DataCollectorAgentPanel />
        </aside>
      </div>
    </div>
  );
} 