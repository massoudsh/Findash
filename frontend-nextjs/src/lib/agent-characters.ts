/**
 * Agent characters for the platform.
 * Each agent has a persona used across Command Center, Risk, Portfolio, Backtesting, etc.
 * Aligned with backend: intelligence_orchestrator (M1–M11).
 */

export const AGENT_IDS = [
  'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11',
] as const;

export type AgentId = typeof AGENT_IDS[number];

export interface AgentCharacter {
  id: AgentId;
  name: string;
  shortName: string;
  tagline: string;
  description: string;
  /** Lucide icon name for UI */
  icon: string;
  /** Tailwind classes for avatar ring / accent (e.g. ring-emerald-500/40) */
  colorClass: string;
  /** Emoji used as avatar fallback or flair */
  emoji: string;
  /** Pages/sections this agent is responsible for */
  responsibleFor: string[];
}

export const AGENT_CHARACTERS: Record<AgentId, AgentCharacter> = {
  M1: {
    id: 'M1',
    name: 'Data Collection Agent',
    shortName: 'Nexus',
    tagline: 'I keep every feed flowing',
    description: 'Collects and normalizes market data, news, and alternative data from all sources.',
    icon: 'Database',
    colorClass: 'ring-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300',
    emoji: '📡',
    responsibleFor: ['Market Data', 'Data Explorer', 'Command Center (data pipeline)'],
  },
  M2: {
    id: 'M2',
    name: 'Data Warehouse Agent',
    shortName: 'Vault',
    tagline: 'Your data, organized',
    description: 'Stores, validates, and serves historical and real-time datasets.',
    icon: 'Archive',
    colorClass: 'ring-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-300',
    emoji: '🗄️',
    responsibleFor: ['Data Explorer', 'Exports', 'Historical data'],
  },
  M3: {
    id: 'M3',
    name: 'Real-time Processing Agent',
    shortName: 'Pulse',
    tagline: 'Live data, zero lag',
    description: 'Processes streaming market data and powers live analytics and alerts.',
    icon: 'Activity',
    colorClass: 'ring-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300',
    emoji: '⚡',
    responsibleFor: ['Real-time', 'Live Trading', 'Alerts'],
  },
  M4: {
    id: 'M4',
    name: 'Strategy Agent',
    shortName: 'Atlas',
    tagline: 'Signals that move the market',
    description: 'Generates trading signals and combines strategy execution with backtesting.',
    icon: 'Target',
    colorClass: 'ring-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
    emoji: '🎯',
    responsibleFor: ['Command Center', 'Strategies', 'Trading Bots', 'Signals'],
  },
  M5: {
    id: 'M5',
    name: 'ML Models Agent',
    shortName: 'Neuron',
    tagline: 'Deep learning, deeper edge',
    description: 'Runs prediction, classification, and deep learning models.',
    icon: 'Brain',
    colorClass: 'ring-fuchsia-500/40 bg-fuchsia-500/10 text-fuchsia-700 dark:text-fuchsia-300',
    emoji: '🧠',
    responsibleFor: ['AI Models', 'Training', 'Predictions'],
  },
  M6: {
    id: 'M6',
    name: 'Risk Management Agent',
    shortName: 'Guardian',
    tagline: 'Risk under control',
    description: 'Assesses risk, position sizing, VaR, and portfolio compliance.',
    icon: 'Shield',
    colorClass: 'ring-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300',
    emoji: '🛡️',
    responsibleFor: ['Risk Dashboard', 'Portfolio risk', 'Compliance'],
  },
  M7: {
    id: 'M7',
    name: 'Price Prediction Agent',
    shortName: 'Oracle',
    tagline: 'Where price goes next',
    description: 'Time-series forecasting and price prediction models.',
    icon: 'TrendingUp',
    colorClass: 'ring-cyan-500/40 bg-cyan-500/10 text-cyan-700 dark:text-cyan-300',
    emoji: '🔮',
    responsibleFor: ['Predictions', 'Technical analysis', 'Forecasts'],
  },
  M8: {
    id: 'M8',
    name: 'Paper Trading Agent',
    shortName: 'Shadow',
    tagline: 'Practice without pressure',
    description: 'Simulates execution and tracks paper portfolio performance.',
    icon: 'Copy',
    colorClass: 'ring-slate-500/40 bg-slate-500/10 text-slate-700 dark:text-slate-300',
    emoji: '📋',
    responsibleFor: ['Paper Trading', 'Portfolio (sim)', 'Execution sim'],
  },
  M9: {
    id: 'M9',
    name: 'Market Sentiment Agent',
    shortName: 'Echo',
    tagline: 'What the crowd feels',
    description: 'Analyzes news and social sentiment for assets and themes.',
    icon: 'MessageSquare',
    colorClass: 'ring-pink-500/40 bg-pink-500/10 text-pink-700 dark:text-pink-300',
    emoji: '💬',
    responsibleFor: ['Social', 'Sentiment', 'Command Center (sentiment panel)'],
  },
  M10: {
    id: 'M10',
    name: 'Backtesting Agent',
    shortName: 'Chronicle',
    tagline: 'History repeats, we measure it',
    description: 'Runs historical backtests and strategy validation.',
    icon: 'History',
    colorClass: 'ring-orange-500/40 bg-orange-500/10 text-orange-700 dark:text-orange-300',
    emoji: '📜',
    responsibleFor: ['Backtesting', 'Strategy validation'],
  },
  M11: {
    id: 'M11',
    name: 'Visualization Agent',
    shortName: 'Lens',
    tagline: 'See the full picture',
    description: 'Powers charts, dashboards, and AI-powered report insights.',
    icon: 'BarChart3',
    colorClass: 'ring-indigo-500/40 bg-indigo-500/10 text-indigo-700 dark:text-indigo-300',
    emoji: '📊',
    responsibleFor: ['Reports', 'Visualization', 'Command Center (insights)', 'Dashboards'],
  },
};

export function getAgentCharacter(id: AgentId): AgentCharacter {
  return AGENT_CHARACTERS[id];
}

export function getAgentsForPage(page: string): AgentId[] {
  const pageLower = page.toLowerCase();
  const out: AgentId[] = [];
  (AGENT_IDS as unknown as AgentId[]).forEach((id) => {
    const a = AGENT_CHARACTERS[id];
    if (a.responsibleFor.some((r) => r.toLowerCase().includes(pageLower) || pageLower.includes(r.toLowerCase().split(' ')[0])))
      out.push(id);
  });
  return out.length ? out : [];
}
