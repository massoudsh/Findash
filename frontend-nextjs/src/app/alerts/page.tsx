import { AlertsPanel } from '@/components/alerts/alerts-panel';

export default function AlertsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">هشدار قیمت</h1>
        <p className="text-muted-foreground">
          قیمت هدف تعیین کن و وقتی دارایی از آستانه عبور کرد اعلان دریافت کن.
        </p>
      </div>
      <AlertsPanel />
    </div>
  );
}
