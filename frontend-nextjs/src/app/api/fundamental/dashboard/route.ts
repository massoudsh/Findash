import { NextRequest } from "next/server";

export async function GET(request: NextRequest) {
  const dashboardData = {
    timestamp: new Date().toISOString(),
    total_assets: 5,
    market_summary: {
      bullish_assets: 3,
      bearish_assets: 1,
      neutral_assets: 1,
      average_confidence: 0.78
    },
    top_signals: [
      {
        asset: "BTC-USD",
        signal_type: "Whale Activity", 
        strength: 8.5,
        description: "Major accumulation detected"
      }
    ]
  };
  
  return Response.json(dashboardData);
}
