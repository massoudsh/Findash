import axios from 'axios';

/** Backend base URL. Use NEXT_PUBLIC_API_URL (no /api suffix), e.g. http://localhost:8000 or http://localhost:8011. */
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Portfolio endpoints
export const getPortfolios = () => api.get('/portfolios/');
export const getPortfolio = (id: number) => api.get(`/portfolios/${id}/`);
export const createPortfolio = (data: any) => api.post('/portfolios/', data);
export const updatePortfolio = (id: number, data: any) => api.put(`/portfolios/${id}/`, data);
export const deletePortfolio = (id: number) => api.delete(`/portfolios/${id}/`);

// Position endpoints
export const getPositions = (portfolioId: number | string) => api.get(`/portfolios/${portfolioId}/positions`);
export const createPosition = (portfolioId: number, data: any) => api.post(`/portfolios/${portfolioId}/positions/`, data);
export const updatePosition = (portfolioId: number, positionId: number, data: any) => 
  api.put(`/portfolios/${portfolioId}/positions/${positionId}/`, data);
export const deletePosition = (portfolioId: number, positionId: number) => 
  api.delete(`/portfolios/${portfolioId}/positions/${positionId}/`);

// Trade endpoints
export const getTrades = (portfolioId: number) => api.get(`/portfolios/${portfolioId}/trades/`);
export const createTrade = (portfolioId: number, data: any) => api.post(`/portfolios/${portfolioId}/trades/`, data);

// Strategy endpoints
export const getStrategies = () => api.get('/strategies/');
export const getStrategy = (id: number) => api.get(`/strategies/${id}/`);
export const createStrategy = (data: any) => api.post('/strategies/', data);
export const updateStrategy = (id: number, data: any) => api.put(`/strategies/${id}/`, data);
export const deleteStrategy = (id: number) => api.delete(`/strategies/${id}/`);
export const backtestStrategy = (id: number, data: any) => api.post(`/strategies/${id}/backtest/`, data);

// Strategy Performance endpoints
export const getStrategyPerformance = (strategyId: number) => api.get(`/strategies/${strategyId}/performance/`);

// --- New Model Registry Endpoint ---
export const getModels = () => api.get('/models');

// --- New Task and Optimization Endpoints ---

/**
 * Polls the task status endpoint until the task is complete or times out.
 * @param taskId The ID of the task to poll.
 * @returns The result of the completed task.
 */
export async function pollTaskStatus(taskId: string): Promise<any> {
  let attempts = 0;
  const maxAttempts = 30; // 30 attempts * 2 seconds = 1 minute timeout
  const interval = 2000; // 2 seconds

  while (attempts < maxAttempts) {
    const response = await api.get(`/tasks/${taskId}`);
    const task = response.data;

    if (task.status === 'SUCCESS') {
      return task.result;
    }

    if (task.status === 'FAILURE') {
      throw new Error(`Task failed: ${task.result}`);
    }

    await new Promise(resolve => setTimeout(resolve, interval));
    attempts++;
  }

  throw new Error('Task polling timed out.');
}

/**
 * Triggers a portfolio optimization task and polls for its result.
 * @param symbols - A list of asset symbols.
 * @param startDate - The start date for historical data (YYYY-MM-DD).
 * @param endDate - The end date for historical data (YYYY-MM-DD).
 * @returns The portfolio optimization results.
 */
export async function runPortfolioOptimization(symbols: string[], startDate: string, endDate: string) {
  const response = await api.post('/portfolio/optimize', {
    symbols: symbols,
    start_date: startDate,
    end_date: endDate,
  });
  
  const initialTask = response.data;
  if (!initialTask.task_id) {
    throw new Error("Backend did not return a task_id");
  }

  return pollTaskStatus(initialTask.task_id);
}

/**
 * Triggers a backtesting task and polls for its result.
 * @param symbol - The asset symbol to backtest.
 * @param startDate - The start date for the backtest (YYYY-MM-DD).
 * @param endDate - The end date for the backtest (YYYY-MM-DD).
 * @param initialCapital - The starting capital for the backtest.
 * @returns The backtesting results, including daily portfolio history.
 */
export async function runBacktest(symbol: string, startDate: string, endDate: string, initialCapital: number) {
  const response = await api.post('/strategies/backtest', {
    symbol,
    start_date: startDate,
    end_date: endDate,
    initial_capital: initialCapital,
  });

  const initialTask = response.data;
  if (!initialTask.task_id) {
    throw new Error("Backend did not return a task_id for backtest");
  }

  return pollTaskStatus(initialTask.task_id);
}

/**
 * Triggers a data ingestion task.
 * @param symbol - The asset symbol to ingest data for.
 * @param startDate - The start date for data ingestion (YYYY-MM-DD).
 * @param endDate - The end date for data ingestion (YYYY-MM-DD).
 * @returns The final status of the task.
 */
export async function ingestData(symbol: string, startDate: string, endDate: string) {
  const response = await api.post('/data/ingest', {
    symbol,
    start_date: startDate,
    end_date: endDate,
  });

  const initialTask = response.data;
  if (!initialTask.task_id) {
    throw new Error("Backend did not return a task_id for ingestion");
  }

  return pollTaskStatus(initialTask.task_id);
}

/**
 * Triggers a model training task.
 * @param symbol - The asset symbol to train a model for.
 * @returns The final status of the task.
 */
export async function trainModel(symbol: string) {
  const response = await api.post('/models/train', { symbol });

  const initialTask = response.data;
  if (!initialTask.task_id) {
    throw new Error("Backend did not return a task_id for training");
  }

  return pollTaskStatus(initialTask.task_id);
}

// Market Data endpoints
export const getTrendingAssets = (symbols: string[]) =>
  api.get('/market/batch', { params: { symbols } });

// News endpoints
export const getNewsArticles = (symbol: string, limit = 10, minSentiment?: number) =>
  api.get(`/news/articles/${symbol}`, {
    params: { limit, ...(minSentiment !== undefined ? { min_sentiment: minSentiment } : {}) },
  });

// Error handling interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Handle specific error cases
      switch (error.response.status) {
        case 401:
          // Handle unauthorized
          console.error('Unauthorized access');
          break;
        case 403:
          // Handle forbidden
          console.error('Forbidden access');
          break;
        case 404:
          // Handle not found
          console.error('Resource not found');
          break;
        case 500:
          // Handle server error
          console.error('Server error');
          break;
        default:
          // Handle other errors
          console.error('API error:', error.response.data);
          break;
      }
    }
    return Promise.reject(error);
  }
);

export default api; 