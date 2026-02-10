import { NextRequest } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  try {
    const { symbol } = await params;
    
    // Return mock data for development
    const mockData = {
      symbol: symbol,
      timestamp: new Date().toISOString(),
      asset_type: 'crypto',
      score: 7.5,
      confidence: 0.85,
      summary: {
        overall_sentiment: 'bullish',
        key_factors: ['Strong on-chain metrics', 'Positive whale activity', 'Institutional interest'],
        signal_count: 12,
        bullish_signals: 8,
        bearish_signals: 4,
      },
      signals: [
        {
          signal_type: 'On-Chain Analysis',
          strength: 8.2,
          confidence: 0.9,
          description: 'Strong network activity with increasing active addresses',
          contributing_factors: ['Active addresses up 15%', 'Transaction volume stable', 'MVRV ratio healthy'],
          timestamp: new Date().toISOString(),
        },
        {
          signal_type: 'Whale Activity',
          strength: 7.8,
          confidence: 0.85,
          description: 'Large holders accumulating, positive net flows',
          contributing_factors: ['Net exchange outflow', 'Whale accumulation score: 8.5', 'Large transactions up 20%'],
          timestamp: new Date().toISOString(),
        },
        {
          signal_type: 'Market Sentiment',
          strength: 6.5,
          confidence: 0.75,
          description: 'Mixed sentiment with bullish bias',
          contributing_factors: ['Social mentions up 10%', 'Fear & Greed index: 65', 'Institutional interest growing'],
          timestamp: new Date().toISOString(),
        },
        {
          signal_type: 'Technical Confluence',
          strength: 7.2,
          confidence: 0.8,
          description: 'Technical indicators align with fundamental outlook',
          contributing_factors: ['Support levels holding', 'Volume profile bullish', 'Momentum indicators positive'],
          timestamp: new Date().toISOString(),
        }
      ],
      on_chain_metrics: {
        active_addresses_24h: 985420,
        transaction_volume_24h: 2.45e9,
        mvrv_ratio: 1.85,
        network_value_to_transactions: 45.2,
        hash_rate: '150 EH/s',
        difficulty_adjustment: '+2.5%'
      },
      whale_metrics: {
        large_transactions_24h: 156,
        whale_accumulation_score: 8.5,
        exchange_inflow: 2500,
        exchange_outflow: 3200,
        net_flow: -700
      }
    };
    
    return Response.json(mockData);
  } catch (error) {
    console.error('Error:', error);
    return Response.json({ error: 'Failed to fetch data' }, { status: 500 });
  }
}
