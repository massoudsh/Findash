/**
 * Asset types and API helpers — Iranian market assets
 */

export type AssetCategory = "gold" | "silver" | "currency" | "real_estate" | "crypto";

export interface Asset {
  symbol: string;
  name_fa: string;
  name_en: string;
  category: AssetCategory;
  unit: string;
  price: number;
  price_toman: number;
  change_24h: number;
  change_percent_24h: number;
  high_24h: number;
  low_24h: number;
  updated_at: string;
  source: string;
}

export interface AssetListResponse {
  assets: Asset[];
  usd_to_toman: number;
  last_updated: string;
}

export interface AssetHistoryPoint {
  timestamp: string;
  close: number;
  open?: number;
  high?: number;
  low?: number;
}

export const CATEGORY_LABELS: Record<AssetCategory, string> = {
  gold:         "طلا و سکه",
  silver:       "نقره",
  currency:     "ارز",
  real_estate:  "مسکن",
  crypto:       "ارز دیجیتال",
};

export const CATEGORY_ICONS: Record<AssetCategory, string> = {
  gold:         "🥇",
  silver:       "🪙",
  currency:     "💵",
  real_estate:  "🏠",
  crypto:       "₿",
};

/** Format number as Persian Toman with commas */
export function formatToman(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)} میلیارد ت`;
  if (n >= 1_000_000)     return `${(n / 1_000_000).toFixed(1)} میلیون ت`;
  return new Intl.NumberFormat("fa-IR").format(Math.round(n)) + " ت";
}

/** Format change percent with sign */
export function formatChange(pct: number): string {
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(2)}%`;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function fetchAssets(category?: AssetCategory): Promise<AssetListResponse> {
  const url = new URL(`${API_BASE}/api/assets`);
  if (category) url.searchParams.set("category", category);
  const res = await fetch(url.toString(), { next: { revalidate: 60 } });
  if (!res.ok) throw new Error("Failed to fetch assets");
  return res.json();
}

export async function fetchAssetHistory(
  symbol: string,
  days = 30
): Promise<AssetHistoryPoint[]> {
  const res = await fetch(
    `${API_BASE}/api/assets/${symbol}/history?days=${days}`,
    { next: { revalidate: 300 } }
  );
  if (!res.ok) return [];
  const data = await res.json();
  return data.data ?? [];
}
