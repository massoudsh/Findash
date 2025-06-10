# Quantum Trading Matrix™
## Advanced Multi-Agent Architecture Blueprint

![Quantum Trading Matrix](https://via.placeholder.com/800x100/0d47a1/ffffff?text=Quantum+Trading+Matrix)

## Executive Summary

The Quantum Trading Matrix™ represents a next-generation trading system that leverages multiple specialized AI agents operating in a coordinated ecosystem. Each agent is responsible for a specific domain of the trading lifecycle, from data collection to risk management. The human trader serves as the central super intelligence, observing agent actions and making final decisions.

---

## System Architecture Blueprint

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUANTUM TRADING MATRIX™ ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                             DATA ACQUISITION LAYER                           │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │ Market Data │    │ Alternative │    │ Regulatory  │    │   Social    │   │
│  │  Streams    │◄──►│  Data Sets  │◄──►│ Filings API │◄──►│   Media     │   │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘   │
│         │                  │                  │                  │          │
│         └──────────────────┼──────────────────┼──────────────────┘          │
│                           ▼                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                   M1 | Data Acquisition Agent                        │   │
│  │                                                                      │   │
│  │  • Adaptive Data Mining    • API Connection Management              │   │
│  │  • Real-time Feed Handling • Data Validation & Verification         │   │
│  └────────────────────────────────┬─────────────────────────────────────┘   │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA PIPELINE LAYER                             │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                   M2 | Data Warehouse Agent                          │   │
│  │                                                                      │   │
│  │  • Data Cleansing      • Feature Engineering       • Schema Design   │   │
│  │  • Data Normalization  • Time Series Alignment     • Query Engine    │   │
│  └────────────────────────────────┬─────────────────────────────────────┘   │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                 M3 | Real-Time Processing Agent                      │   │
│  │                                                                      │   │
│  │  • Stream Processing   • Event Detection      • Drift Detection      │   │
│  │  • Online Learning     • Signal Generation    • Anomaly Detection    │   │
│  └────────────────────────────────┬─────────────────────────────────────┘   │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INTELLIGENCE LAYER                                │
│                                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │    M5 | DL     │  │     M7 |       │  │     M9 |       │                 │
│  │  Models Agent  │  │Price Prediction│  │Market Sentiment│                 │
│  │                │  │     Agent      │  │     Agent      │                 │
│  │• TCN Networks  │  │• Prophet Models│  │• FinBERT NLP   │                 │
│  │• Transformers  │  │• YOLO Patterns │  │• Trend Analysis│  │• News Analysis │                 │
│  │• AutoEncoders  │  │• Trend Analysis│  │• Social Signal │                 │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘                 │
│           │                   │                    │                         │
│           └───────────────────┼────────────────────┘                         │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      M4 | Strategy Agent                             │   │
│  │                                                                      │   │
│  │  • Strategy Selection    • Parameter Optimization   • Alpha Capture  │   │
│  │  • Execution Planning    • Regime Detection         • Signal Fusion  │   │
│  └────────────────────────────────┬─────────────────────────────────────┘   │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RISK CONTROL LAYER                              │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    M6 | Risk Management Agent                        │   │
│  │                                                                      │   │
│  │  • Position Sizing      • VaR Calculation       • Tail Risk Analysis │   │
│  │  • Exposure Monitoring  • Correlation Matrices  • Portfolio Optim.   │   │
│  └────────────────────────────────┬─────────────────────────────────────┘   │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    M8 | Paper Trading Agent                          │   │
│  │                                                                      │   │
│  │  • Strategy Simulation  • Performance Metrics   • Parameter Tuning   │   │
│  │  • Scenario Analysis    • Drawdown Assessment   • Execution Analysis │   │
│  └────────────────────────────────┬─────────────────────────────────────┘   │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VALIDATION & REPORTING LAYER                       │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    M10 | Backtesting Agent                           │   │
│  │                                                                      │   │
│  │  • Historical Validation  • Monte Carlo Testing  • Strategy Ranking  │   │
│  │  • Robustness Analysis    • Walk-Forward Testing • Stat. Significance│   │
│  └────────────────────────────────┬─────────────────────────────────────┘   │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                   M11 | Visualization Agent                          │   │
│  │                                                                      │   │
│  │  • Interactive Dashboards  • Performance Reports  • Risk Heatmaps    │   │
│  │  • Decision Trees          • Execution Analytics  • Pattern Displays │   │
│  └────────────────────────────────┬─────────────────────────────────────┘   │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SUPERINTELLIGENCE NODE                             │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Human Decision Matrix                             │   │
│  │                                                                      │   │
│  │  • Final Trade Authorization   • Risk Tolerance Setting              │   │
│  │  • Agent Feedback Loop         • Strategy Weighting                  │   │
│  │  • System Parameter Control    • Contingency Activation              │   │
│  └────────────────────────────────┬─────────────────────────────────────┘   │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Execution Interface                               │   │
│  │                                                                      │   │
│  │  • Order Management System     • Capital Allocation                  │   │
│  │  • Execution Algorithm Select  • Performance Tracking                │   │
│  │  • Compliance Verification     • Transaction Cost Analysis           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Cross-Agent Communication Matrix

| Agent Interface | Primary Data Flow | Feedback Mechanisms | Integration Protocols |
|----------------|-------------------|---------------------|------------------------|
| **Data → Intelligence** | Normalized market data, Feature vectors | Data quality metrics, Usage statistics | Stream processing pipeline, Feature API |
| **Intelligence → Strategy** | Alpha signals, Pattern detections, Sentiment scores | Signal performance metrics, Confidence intervals | Signal fusion algorithm, Multi-factor model |
| **Strategy → Risk** | Proposed trades, Expected returns, Trading theses | Risk-adjusted returns, VaR exposures | Position sizing API, Risk budget allocation |
| **Risk → Validation** | Approved positions, Risk constraints, Max drawdowns | Compliance validations, Breach notifications | Risk envelope protocol, Regulatory framework |
| **Validation → Human** | Performance dashboards, Backtest results, Execution analytics | Decision annotations, Strategy weightings | Interactive visualization API, Alert system |
| **Human → System** | Authorization decisions, Parameter adjustments, Strategy selections | Feedback ratings, Override logs | Command interface, Configuration management |

## Agent Quantum Entanglement Matrix

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         CROSS-AGENT INTEGRATION                           │
└───────────────────────────────────────────────────────────────────────────┘

  DATA PROCESSING       INTELLIGENCE       EXECUTION         GOVERNANCE
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Data Pipeline  │  │  Alpha Factory  │  │ Execution Suite │  │    Oversight    │
│                 │  │                 │  │                 │  │                 │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │    M1     │◄─┼──┼──┤    M5     │  │  │  │    M6     │◄─┼──┼──┤   Human   │  │
│  │Collection │  │  │  │   Models  │  │  │  │   Risk    │  │  │  │Super Node │  │
│  └─────┬─────┘  │  │  └─────┬─────┘  │  │  └─────┬─────┘  │  │  └─────┬─────┘  │
│        │        │  │        │        │  │        │        │  │        │        │
│  ┌─────▼─────┐  │  │  ┌─────▼─────┐  │  │  ┌─────▼─────┐  │  │  ┌─────▼─────┐  │
│  │    M2     │◄─┼──┼──┤    M7     │  │  │  │    M8     │◄─┼──┼──┤    M11    │  │
│  │ Warehouse │  │  │  │ Prediction │  │  │  │   Paper   │  │  │  │Visulizatn│  │
│  └─────┬─────┘  │  │  └─────┬─────┘  │  │  └─────┬─────┘  │  │  └───────────┘  │
│        │        │  │        │        │  │        │        │  │                 │
│  ┌─────▼─────┐  │  │  ┌─────▼─────┐  │  │  ┌─────▼─────┐  │  │  ┌───────────┐  │
│  │    M3     │◄─┼──┼──┤    M9     │  │  │  │    M10    │◄─┼──┼──┤ Feedback  │  │
│  │ Real-Time │  │  │  │ Sentiment │  │  │  │ Backtest  │  │  │  │   Loop    │  │
│  └─────┬─────┘  │  │  └─────┬─────┘  │  │  └───────────┘  │  │  └─────┬─────┘  │
│        │        │  │        │        │  │                 │  │        │        │
│  ┌─────▼─────┐  │  │  ┌─────▼─────┐  │  │                 │  │        │        │
│  │  Unified  │◄─┼──┼──┤    M4     │◄─┼──┼─────────────────┼──┼────────┘        │
│  │ Data API  │  │  │  │ Strategies │  │  │                 │  │                 │
│  └───────────┘  │  │  └───────────┘  │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Decision Tensor Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     QUANTUM DECISION MATRIX                         │
└─────────────────────────────────────────────────────────────────────┘

   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
   │  Intelligence │    │Risk Assessment│    │  Backtesting  │
   │    Signals    │    │   Metrics     │    │   Results     │
   └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
           │                    │                    │
           ▼                    ▼                    ▼
   ┌─────────────────────────────────────────────────────────┐
   │                                                         │
   │                 NEURAL DECISION ENGINE                  │
   │                                                         │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
   │  │ Technical   │  │ Fundamental │  │ Sentiment   │     │
   │  │ Analysis    │  │ Analysis    │  │ Analysis    │     │
   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
   │         │                │                │            │
   │         └────────────────┼────────────────┘            │
   │                          │                             │
   │  ┌─────────────┐  ┌──────▼──────┐  ┌─────────────┐     │
   │  │ Correlation │  │ Probability │  │ Confidence  │     │
   │  │ Engine      │  │ Matrix      │  │ Scoring     │     │
   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
   │         │                │                │            │
   │         └────────────────┼────────────────┘            │
   │                          │                             │
   │                 ┌────────▼────────┐                    │
   │                 │  Signal Fusion  │                    │
   │                 │    Algorithm    │                    │
   │                 └────────┬────────┘                    │
   │                          │                             │
   └──────────────────────────┼─────────────────────────────┘
                             ▼
                     ┌───────────────┐
                     │    Human      │
                     │  Super Node   │
                     └───────┬───────┘
                             │
                             ▼
                     ┌───────────────┐
                     │   Execution   │
                     │   Platform    │
                     └───────────────┘
```

## System Phase States

| Phase | State | Active Agents | Function |
|-------|-------|--------------|----------|
| **Observation** | Data Collection | M1, M2, M3 | Continuous market data ingestion and processing |
| **Analysis** | Signal Generation | M5, M7, M9 | Multi-modal analysis across time series, patterns, and text |
| **Strategy** | Position Planning | M4, M6 | Strategy selection and risk-adjusted position sizing |
| **Simulation** | Pre-Execution Testing | M8, M10 | Simulated execution and historical validation |
| **Visualization** | Decision Support | M11 | Interactive data presentation and anomaly highlighting |
| **Decision** | Human Approval | Super Node | Final trade authorization and parameter adjustment |
| **Execution** | Order Management | External Broker API | Trade execution and settlement |
| **Feedback** | Performance Analysis | All Agents | Continuous learning and system optimization |

## Technological Tensor Infrastructure

```
┌─────────────────────────────────────────────────────────────────┐
│                  COMPUTATIONAL ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     CONTAINERIZATION LAYER                      │
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │Docker Images│   │Kubernetes   │   │Service Mesh │           │
│  │& Containers │   │Orchestration│   │ Network     │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
└──────────────────────────────┬──────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      COMPUTATIONAL GRID                         │
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ CPU Cluster │   │ GPU Arrays  │   │ Memory Grid │           │
│  │ Processing  │   │ For ML/DL   │   │ Distributed │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
└──────────────────────────────┬──────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SUBSTRATE                          │
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ PostgreSQL  │   │   Redis     │   │Distributed  │           │
│  │  Database   │   │Cache & Queue│   │ File System │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Intercommunication Protocol

Each agent exposes and consumes standardized APIs that enable quantum-secure information exchange:

1. **Data Exchange Protocol (DEP)**: JSON/Avro schema for structured data transmission
2. **Signal Messaging Interface (SMI)**: Pub/sub system for real-time event notifications
3. **Model Parameter Exchange (MPE)**: Secure container for ML model weights and hyperparameters
4. **Execution Directive Format (EDF)**: Standardized trade instruction encoding
5. **Visualization Query Language (VQL)**: Grammar for requesting specific data visualizations
6. **Human-Agent Dialogue System (HADS)**: Natural language interface for human interaction

## Strategic Differentiation Vectors

1. **Multi-modal Intelligence Fusion**: Unlike systems that rely on a single data type, our platform synthesizes time-series, textual, and visual pattern data
   
2. **Quantum Signal Processing**: Our proprietary signal fusion algorithm resolves contradictory indicators using quantum-inspired probability states

3. **Hierarchical Risk Management**: Risk is evaluated at multiple levels: trade, strategy, portfolio, and system-wide

4. **Human-AI Collaborative Matrix**: Optimizes human cognitive resources by focusing decision-making on edge cases and strategic pivots

5. **AI Sentiment Amplification**: FinBERT NLP transforms unstructured financial text into tradable signals with domain-specific understanding

## Advanced Integration Points

- **External API Matrix**: Connects to brokerages, data vendors, and regulatory systems
- **Quantum Security Layer**: Ensures data integrity and system access control
- **Continuous Integration/Deployment Pipeline**: Enables seamless agent updates
- **Federated Learning Network**: Allows distributed model training while preserving data privacy
- **Regulatory Compliance Engine**: Ensures all actions adhere to financial regulations

## Technology Stack

- **Languages**: Python, JavaScript
- **AI/ML**: TensorFlow, PyTorch, Transformers, Prophet, XGBoost
- **Infrastructure**: Docker, Kubernetes, AWS
- **Database**: PostgreSQL, Redis
- **Monitoring**: Prometheus, Grafana

## Investment Opportunity

The Quantum Trading Matrix™ represents the convergence of artificial intelligence, sophisticated software engineering, and human expertise, creating an unprecedented trading intelligence ecosystem with:

- **Reduced model risk** through multi-agent consensus
- **Enhanced adaptability** via modular architecture
- **Improved explainability** through human-AI collaboration
- **Competitive edge** from multi-modal signal fusion

## Marketing Strategy & Go-To-Market Plan

### Executive Marketing Summary

The Quantum Trading Matrix™ represents a revolutionary leap in algorithmic trading technology, combining advanced AI agents with human expertise. Our marketing strategy focuses on positioning the platform as the premier solution for institutional investors, hedge funds, and sophisticated traders seeking a competitive edge in today's complex markets.

### Target Market Segments

1. **Primary Markets**
   - Institutional Investment Firms
   - Hedge Funds & Proprietary Trading Desks
   - Quantitative Trading Teams
   - Family Offices & Wealth Management Firms

2. **Secondary Markets**
   - FinTech Innovation Departments
   - Academic Research Institutions
   - Cryptocurrency Trading Firms
   - High-Net-Worth Individual Traders

### Unique Value Proposition

```
┌─────────────────────────────────────────────────────────────┐
│             QUANTUM ADVANTAGE MATRIX                        │
├─────────────────┬───────────────────────────────────────────┤
│ Intelligence    │ • Multi-agent AI architecture             │
│                 │ • Quantum-inspired signal processing      │
├─────────────────┼───────────────────────────────────────────┤
│ Risk Control    │ • Hierarchical risk management            │
│                 │ • Real-time portfolio optimization        │
├─────────────────┼───────────────────────────────────────────┤
│ Adaptability    │ • Self-learning algorithms                │
│                 │ • Dynamic strategy adjustment             │
├─────────────────┼───────────────────────────────────────────┤
│ Human Override  │ • Intuitive control interface             │
│                 │ • Expert supervision capabilities         │
└─────────────────┴───────────────────────────────────────────┘
```

### Marketing Channels & Tactics

1. **Digital Presence**
   - Professional website with interactive demos
   - Technical blog showcasing research & insights
   - LinkedIn company page & thought leadership
   - YouTube channel with educational content
   - Regular webinars & virtual demonstrations

2. **Industry Events**
   - QuantCon & similar quant trading conferences
   - AI & Machine Learning in Finance summits
   - Algorithmic Trading exhibitions
   - Financial technology innovation forums

3. **Content Marketing**
   - White papers on agent architecture
   - Case studies & performance metrics
   - Technical documentation & tutorials
   - Market research & analysis reports

4. **Partnership Program**
   - Data provider collaborations
   - Broker integration partnerships
   - Academic research partnerships
   - Technology vendor alliances

### Lead Generation Strategy

1. **Inbound Marketing**
   - SEO-optimized technical content
   - Gated premium research papers
   - Newsletter subscription program
   - Free trial registration system

2. **Outbound Initiatives**
   - Direct outreach to institutional investors
   - LinkedIn InMail campaigns
   - Industry-specific email marketing
   - Targeted advertising on financial platforms

### Sales Process

```
┌─────────────────────────────────────────────────────────────┐
│                 SALES PIPELINE MATRIX                       │
│                                                            │
│  Discovery → Demo → Technical Review → Pilot → Deployment   │
│                                                            │
│  • Initial consultation    • Custom solution design        │
│  • Platform demonstration  • Risk assessment               │
│  • Technical deep dive    • Implementation planning        │
│  • Pilot program setup    • Training & support            │
└─────────────────────────────────────────────────────────────┘
```

### Pricing Strategy

1. **Enterprise Tier**
   - Full platform access
   - Custom integration support
   - Dedicated account management
   - 24/7 technical support

2. **Professional Tier**
   - Core functionality access
   - Standard integrations
   - Business hours support
   - Regular updates & maintenance

3. **Academic License**
   - Research-focused access
   - Limited trading capabilities
   - Community support
   - Educational resources

### Marketing KPIs & Metrics

1. **Acquisition Metrics**
   - Lead generation rate
   - Demo request conversion
   - Trial activation rate
   - Sales pipeline velocity

2. **Engagement Metrics**
   - Platform usage statistics
   - Feature adoption rates
   - Support ticket resolution
   - Client satisfaction scores

3. **Revenue Metrics**
   - Monthly recurring revenue
   - Customer lifetime value
   - Churn rate
   - Expansion revenue

### Implementation Timeline

```
Q1: Launch Phase
├── Website & digital presence setup
├── Initial content creation
├── Partnership program initiation
└── Beta client onboarding

Q2: Growth Phase
├── Event participation
├── Content marketing expansion
├── Case study development
└── Sales team expansion

Q3: Optimization Phase
├── Channel performance analysis
├── Strategy refinement
├── Partnership expansion
└── Product feedback integration

Q4: Scale Phase
├── International market entry
├── Enterprise client focus
├── Advanced feature rollout
└── Community building
```

### Budget Allocation

1. **Digital Marketing**: 30%
   - Website development & maintenance
   - Content creation & distribution
   - Digital advertising
   - SEO & analytics

2. **Events & PR**: 25%
   - Conference participation
   - Industry events
   - Press relations
   - Sponsorships

3. **Sales & Support**: 35%
   - Sales team resources
   - Technical support
   - Training materials
   - Client success

4. **Research & Development**: 10%
   - Market research
   - Product enhancement
   - Competitive analysis
   - User experience studies

### Risk Mitigation

1. **Market Risks**
   - Continuous market monitoring
   - Agile strategy adjustment
   - Competitive analysis
   - Innovation pipeline

2. **Technical Risks**
   - Robust testing protocols
   - Security audits
   - Compliance reviews
   - Performance monitoring

3. **Adoption Risks**
   - User feedback integration
   - Training programs
   - Support resources
   - Success metrics tracking

### Success Metrics

```
┌─────────────────────────────────────────────────────────────┐
│                 SUCCESS METRIC MATRIX                       │
├─────────────────┬───────────────────────────────────────────┤
│ Market          │ • Market share growth                     │
│ Performance     │ • Brand recognition                       │
├─────────────────┼───────────────────────────────────────────┤
│ Client          │ • Adoption rate                           │
│ Success         │ • Satisfaction scores                     │
├─────────────────┼───────────────────────────────────────────┤
│ Financial       │ • Revenue growth                          │
│ Impact          │ • Profitability metrics                   │
├─────────────────┼───────────────────────────────────────────┤
│ Technical       │ • Platform stability                      │
│ Excellence      │ • Innovation metrics                      │
└─────────────────┴───────────────────────────────────────────┘
```

---

© 2023 Quantum Trading Matrix. All Rights Reserved.
For more information, contact: info@quantumtradingmatrix.com 

# Project Root

## New: src/ Directory for Core Python Code

To improve modularity and maintainability, please place all core reusable Python modules in a top-level `src/` directory. If you refactor or add new core logic, use `src/` as the main location. 

## Environment Variables

Copy `.env.example` to `.env` and fill in the required values. Use [python-dotenv](https://github.com/theskumar/python-dotenv) to load environment variables in local development. 

## Makefile

Use the provided `Makefile` for common development tasks:
- `make install` — Install all dependencies
- `make lint` — Run code quality checks
- `make format` — Auto-format code
- `make coverage` — Run tests with coverage
- `make test` — Run tests
- `make frontend` — Build the Next.js frontend 