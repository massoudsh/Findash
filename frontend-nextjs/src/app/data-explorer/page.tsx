import { DataExplorer } from "@/components/data/data-explorer";
import { DataExport } from "@/components/ui/data-export";

export default function DataExplorerPage() {
  return (
    <div className="container mx-auto py-6">
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
  );
} 