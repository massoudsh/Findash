interface GrafanaDashboardProps {
  url: string;
  title?: string;
  height?: number;
}

export function GrafanaDashboard({ url, title = 'Grafana Dashboard', height = 600 }: GrafanaDashboardProps) {
  return (
    <div className="rounded border bg-background">
      {title && <div className="font-semibold px-4 py-2 border-b">{title}</div>}
      <iframe
        src={url}
        width="100%"
        height={height}
        frameBorder={0}
        allowFullScreen
        className="w-full"
      />
    </div>
  );
}
