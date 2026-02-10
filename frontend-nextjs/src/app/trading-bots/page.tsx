'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface TradingBot {
  id: string;
  name: string;
  strategy: string;
  status: 'active' | 'paused' | 'stopped';
  performance: {
    total_trades: number;
    win_rate: number;
    total_pnl: number;
  };
  created_at: string;
}

export default function TradingBotsPage() {
  const [bots, setBots] = useState<TradingBot[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newBot, setNewBot] = useState({ name: '', strategy: 'momentum' });

  useEffect(() => {
    const fetchBots = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/trading-bots/`);
        setBots(response.data || []);
      } catch (error) {
        console.error('Error fetching bots:', error);
        // Mock data for demo
        setBots([
          {
            id: '1',
            name: 'Momentum Bot',
            strategy: 'momentum',
            status: 'active',
            performance: { total_trades: 45, win_rate: 0.67, total_pnl: 1250.50 },
            created_at: new Date().toISOString(),
          },
          {
            id: '2',
            name: 'Mean Reversion Bot',
            strategy: 'mean_reversion',
            status: 'paused',
            performance: { total_trades: 32, win_rate: 0.59, total_pnl: 890.25 },
            created_at: new Date().toISOString(),
          },
        ]);
      } finally {
        setLoading(false);
      }
    };
    fetchBots();
  }, []);

  const createBot = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/trading-bots/`, newBot);
      setBots([...bots, response.data]);
      setShowCreateModal(false);
      setNewBot({ name: '', strategy: 'momentum' });
    } catch (error) {
      console.error('Error creating bot:', error);
    }
  };

  const toggleBotStatus = async (botId: string, currentStatus: string) => {
    try {
      const action = currentStatus === 'active' ? 'pause' : 'start';
      await axios.post(`${API_BASE_URL}/api/trading-bots/${botId}/${action}`);
      setBots(bots.map(bot => 
        bot.id === botId 
          ? { ...bot, status: action === 'pause' ? 'paused' : 'active' }
          : bot
      ));
    } catch (error) {
      console.error('Error toggling bot status:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div>Loading trading bots...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">🤖 Trading Bots</h1>
          <p className="text-muted-foreground">
            Create and manage automated trading bots
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          + Create Bot
        </button>
      </div>

      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowCreateModal(false)}>
          <div className="bg-white rounded-lg p-6 w-full max-w-md" onClick={(e) => e.stopPropagation()}>
            <h2 className="text-2xl font-bold mb-4">Create Trading Bot</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Bot Name</label>
                <input
                  type="text"
                  value={newBot.name}
                  onChange={(e) => setNewBot({ ...newBot, name: e.target.value })}
                  placeholder="Enter bot name"
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Strategy</label>
                <select
                  value={newBot.strategy}
                  onChange={(e) => setNewBot({ ...newBot, strategy: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                >
                  <option value="momentum">Momentum</option>
                  <option value="mean_reversion">Mean Reversion</option>
                  <option value="arbitrage">Arbitrage</option>
                  <option value="scalping">Scalping</option>
                </select>
              </div>
              <div className="flex gap-2 justify-end">
                <button
                  onClick={() => setShowCreateModal(false)}
                  className="px-4 py-2 bg-gray-200 rounded-lg hover:bg-gray-300"
                >
                  Cancel
                </button>
                <button
                  onClick={createBot}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Create
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {bots.map((bot) => (
          <div key={bot.id} className="bg-white rounded-lg shadow p-6">
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-xl font-semibold">{bot.name}</h3>
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                bot.status === 'active' ? 'bg-green-100 text-green-800' :
                bot.status === 'paused' ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {bot.status}
              </span>
            </div>
            <div className="space-y-2 mb-4">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Strategy:</span>
                <span className="font-medium">{bot.strategy}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Total Trades:</span>
                <span className="font-medium">{bot.performance.total_trades}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Win Rate:</span>
                <span className="font-medium">{(bot.performance.win_rate * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Total P&L:</span>
                <span className={`font-medium ${bot.performance.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  ${bot.performance.total_pnl.toFixed(2)}
                </span>
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => toggleBotStatus(bot.id, bot.status)}
                className={`flex-1 px-3 py-2 rounded text-sm font-medium ${
                  bot.status === 'active' 
                    ? 'bg-yellow-500 text-white hover:bg-yellow-600' 
                    : 'bg-green-500 text-white hover:bg-green-600'
                }`}
              >
                {bot.status === 'active' ? '⏸ Pause' : '▶ Start'}
              </button>
              <button
                onClick={() => toggleBotStatus(bot.id, 'stopped')}
                className="px-3 py-2 bg-red-500 text-white rounded text-sm font-medium hover:bg-red-600"
              >
                ⏹ Stop
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

