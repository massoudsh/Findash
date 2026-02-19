# Changelog

All notable changes to the Octopus Trading Platform (Findash) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Full development timeline (zero to now) with structured logging parameters:** [docs/DEVELOPMENT-LOG.md](docs/DEVELOPMENT-LOG.md).

## [Unreleased]

### Added
- Comprehensive GitHub Wiki documentation
- Modern glassmorphism UI card components
- Dual sidebar navigation (left + right)
- Wiki publish automation script

### Changed
- Enhanced card components with variants (glass, gradient, elevated, bordered)
- Reorganized navigation tabs into logical groups
- Improved dashboard styling with modern design patterns

## [3.3.0] - 2026-01-15

### Added
- Trading Bots module with automated strategy execution
- Enhanced Backtesting engine with Monte Carlo simulation
- Unified market data workflow
- Real-time WebSocket improvements
- Portfolio optimization algorithms

### Changed
- Refactored intelligence orchestrator for 11 AI agents
- Improved risk management calculations
- Enhanced caching with Redis integration

### Fixed
- Database connection handling in development mode
- API exception handlers for better error responses
- Cache decorator implementation

## [3.2.0] - 2025-12-01

### Added
- AI Agents dashboard for monitoring all 11 agents
- Advanced risk metrics (VaR, Sharpe, Sortino)
- Options trading engine integration
- Sentiment analysis from news and social media

### Changed
- Migrated to Next.js 14 App Router
- Updated Tailwind CSS configuration
- Improved TypeScript type definitions

### Fixed
- WebSocket connection stability
- Portfolio calculation accuracy
- Authentication token refresh logic

## [3.1.0] - 2025-11-01

### Added
- Market data streaming via Kafka
- Celery task monitoring with Flower
- Prometheus metrics exporter
- Grafana dashboards for trading metrics

### Changed
- Replaced Kafka with Redis Streams for simpler deployment
- Optimized database queries with proper indexing
- Enhanced logging configuration

## [3.0.0] - 2025-10-01

### Added
- Complete platform rewrite with microservices architecture
- 11 AI agents for intelligent trading
- Intelligence Orchestrator for agent coordination
- TimescaleDB for time-series market data
- Professional authentication with JWT/OAuth2

### Changed
- New modern frontend with glassmorphism design
- Restructured backend with FastAPI
- Implemented proper separation of concerns

### Breaking Changes
- New database schema (migration required)
- Updated API endpoints (v1 → v2)
- Changed configuration format

## [2.0.0] - 2025-08-01

### Added
- Initial trading platform features
- Basic portfolio management
- Market data integration
- User authentication

## [1.0.0] - 2025-06-01

### Added
- Initial release
- Basic project structure
- Core trading functionality

---

[Unreleased]: https://github.com/massoudsh/Findash/compare/v3.3.0...HEAD
[3.3.0]: https://github.com/massoudsh/Findash/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/massoudsh/Findash/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/massoudsh/Findash/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/massoudsh/Findash/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/massoudsh/Findash/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/massoudsh/Findash/releases/tag/v1.0.0
