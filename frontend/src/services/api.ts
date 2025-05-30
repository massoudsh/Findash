import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

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
export const getPositions = (portfolioId: number) => api.get(`/portfolios/${portfolioId}/positions/`);
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

// Error handling interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Handle specific error cases
      switch (error.response.status) {
        case 401:
          // Handle unauthorized
          break;
        case 403:
          // Handle forbidden
          break;
        case 404:
          // Handle not found
          break;
        case 500:
          // Handle server error
          break;
        default:
          // Handle other errors
          break;
      }
    }
    return Promise.reject(error);
  }
);

export default api; 