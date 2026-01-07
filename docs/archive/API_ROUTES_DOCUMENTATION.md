# API Routes Documentation

## Overview

This document provides a comprehensive guide to all API routes in the Octopus Trading Platform.

## Route Organization

### Authentication & Authorization
- `/api/auth/*` - Professional Authentication (`professional_auth.py`)
- `/auth/*` - Legacy Authentication (`auth.py`)

### Market Data APIs

#### Professional Market Data
- **Prefix**: `/api/market-data`
- **Router**: `professional_market_data.py`
- **Purpose**: Professional-grade real-time and historical market data
- **Features**: Real-time quotes, historical data, technical indicators

#### Market Data Workflow
- **Prefix**: `/api/market-data/workflow`
- **Router**: `market_data_workflow.py`
- **Purpose**: Workflow management for fetching market data from free APIs
- **Features**: Batch fetching, async tasks, source status, historical data

#### Legacy Market Data (V1)
- **Prefix**: `/api/v1/market-data`
- **Router**: `market_data.py` (from routes)
- **Purpose**: Legacy API for backward compatibility

#### Simple Real Market Data
- **Prefix**: `/api` (specific endpoints)
- **Router**: `simple_real_data.py`
- **Purpose**: Simple, lightweight market data endpoints

### Real-Time Data
- **Prefix**: `/api/v1/realtime`
- **Router**: `realtime.py` (from routes)
- **Purpose**: Real-time data streaming and updates

### Machine Learning
- **Prefix**: `/api/v1/ml`
- **Router**: `ml_models.py` (from routes)
- **Purpose**: ML model management and predictions

### Portfolio Management
- **Prefix**: `/portfolios`
- **Router**: `portfolios.py` (from routes)
- **Purpose**: Portfolio CRUD operations

### Agent Monitoring
- **Prefix**: `/api/agents`
- **Router**: `agents.py`
- **Purpose**: AI agent status, logs, and decisions

- **Prefix**: `/api/agents/v2`
- **Router**: `agents_v2.py`
- **Purpose**: Alternative/backup agent monitoring API

### Wallet & Funding
- **Prefix**: `/api/wallet`
- **Router**: `wallet.py`
- **Purpose**: Multi-currency balances, transactions, bank accounts

### Security & Access Control
- **Prefix**: `/api/security`
- **Router**: `security.py`
- **Purpose**: API keys, sessions, trading permissions

### Market Scenarios
- **Prefix**: `/api/scenarios`
- **Router**: `scenarios.py`
- **Purpose**: Market regime scenario analysis

### Risk Management
- **Prefix**: `/api/risk`
- **Router**: `risk.py`
- **Purpose**: Risk metrics and analysis

### LLM & AI Analytics
- **Prefix**: `/llm`
- **Router**: `llm_simple.py`
- **Purpose**: Language model operations

### Real Data Sources
- **Prefix**: `/api` (specific endpoints)
- **Routers**: 
  - `macro_data.py` - Macro economic data
  - `onchain_data.py` - On-chain crypto data
  - `social_data.py` - Social sentiment data

### WebSocket Endpoints

#### Main WebSocket
- **Path**: `/ws`
- **Type**: Direct WebSocket endpoint
- **Purpose**: Real-time market data streaming
- **Implementation**: `main_refactored.py` (direct endpoint)

#### WebSocket API (V1)
- **Prefix**: `/api/v1/websocket`
- **Router**: `websocket.py` (from routes)
- **Purpose**: Legacy WebSocket API

#### WebSocket Real-time
- **Prefix**: `/api/ws`
- **Router**: `websocket_realtime.py`
- **Purpose**: Real-time agent status and wallet transaction updates

### Comprehensive API
- **Prefix**: `/api`
- **Router**: `comprehensive_api.py`
- **Purpose**: Comprehensive trading operations and analytics

## Route Conflicts Resolution

### Market Data Routes
All market data routes are now properly separated:
- `/api/market-data` - Professional API
- `/api/market-data/workflow` - Workflow management
- `/api/v1/market-data` - Legacy V1 API
- `/api/real-market-data` - Simple endpoints

### Agent Routes
- `/api/agents` - Main agent monitoring
- `/api/agents/v2` - Alternative/backup API

### WebSocket Routes
- `/ws` - Main WebSocket (direct)
- `/api/v1/websocket` - Legacy API
- `/api/ws` - Real-time updates

## Best Practices

1. **Use Professional APIs**: Prefer `/api/market-data` over legacy routes
2. **Workflow API**: Use `/api/market-data/workflow` for batch operations
3. **WebSocket**: Use `/ws` for main real-time streaming
4. **Versioning**: V1 routes are for backward compatibility only

## Migration Guide

### From Legacy to Professional APIs

**Old**: `/api/v1/market-data/current/{symbol}`
**New**: `/api/market-data/quote/{symbol}`

**Old**: `/api/v1/market-data/historical/{symbol}`
**New**: `/api/market-data/historical/{symbol}`

### From Simple to Workflow API

**Old**: `/api/real-market-data`
**New**: `/api/market-data/workflow/fetch/{symbol}`

## Authentication

Most endpoints require JWT authentication:
```
Authorization: Bearer <token>
```

Exceptions:
- `/health` - Public health check
- `/docs` - API documentation
- `/` - Root endpoint

## Rate Limiting

All authenticated endpoints have rate limiting:
- Default: 30 requests per minute
- Some endpoints may have custom limits

Check response headers:
- `X-RateLimit-Limit`: Maximum requests
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset time

