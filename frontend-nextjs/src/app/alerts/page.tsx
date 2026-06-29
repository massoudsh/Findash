import { AlertsPanel } from '@/components/alerts/alerts-panel';

export default function AlertsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Price Alerts</h1>
        <p className="text-muted-foreground">
          Set target prices and get notified when assets cross your thresholds.
        </p>
      </div>
      <AlertsPanel />
    </div>
  );
}
