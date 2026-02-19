# Agent–User Decision Workflow

**From market updates to visualizations and reports** — a step-by-step view of how Octopus agents and users collaborate for investment decisions (Aladdin-style decision support).

---

## Overview

The platform combines **continuous data ingestion**, **agent-driven analytics**, and **human decisions** in a single loop: data flows in → agents analyze and recommend → the user decides → execution and reporting close the loop.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MARKET DATA  →  ENRICHMENT  →  SIGNALS & RISK  →  YOUR DECISION  →  EXECUTION   │
│       ↑              ↑                ↑                ↑                ↑       │
│     M1,M3,M9       M2,M5,M7         M4,M6           👤 You           M8,M11     │
│                                                          ↓                       │
│  REPORTS & VISUALIZATION  ←  LENS (M11)  ←  Results & attribution                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. End-to-end pipeline (high level)

```mermaid
flowchart LR
    subgraph SOURCES["📡 Sources"]
        MKT[Markets]
        NEWS[News & Social]
        ALT[Alternative Data]
    end

    subgraph INGEST["🔄 Ingest & Store"]
        M1[Nexus M1]
        M2[Vault M2]
        M3[Pulse M3]
        M9[Echo M9]
    end

    subgraph ANALYZE["🧠 Analyze"]
        M5[Neuron M5]
        M7[Oracle M7]
        M4[Atlas M4]
        M6[Guardian M6]
    end

    subgraph DECIDE["👤 Decide"]
        USER[Trader]
    end

    subgraph EXECUTE["📤 Execute & Report"]
        M8[Shadow M8]
        M10[Chronicle M10]
        M11[Lens M11]
    end

    MKT & NEWS & ALT --> M1
    M1 --> M2
    M2 --> M3
    M1 --> M9
    M3 --> M5 & M7
    M9 --> M4
    M5 & M7 --> M4
    M4 --> M6
    M6 --> USER
    USER --> M8 & M10
    M8 --> M11
    M10 --> M11
    M11 --> REPORTS[Reports & Dashboards]
```

---

## 2. Decision workflow by phase (swimlane)

```mermaid
flowchart TB
    subgraph Phase1["Phase 1: Market updates"]
        A1[📡 Nexus M1: Ingest prices, news, alt data]
        A2[🗄️ Vault M2: Store & validate]
        A3[⚡ Pulse M3: Stream to platform]
        A4[💬 Echo M9: Sentiment scores]
        A1 --> A2 --> A3
        A1 --> A4
    end

    subgraph Phase2["Phase 2: Analytics & signals"]
        B1[🧠 Neuron M5: ML predictions]
        B2[🔮 Oracle M7: Price forecasts]
        B3[🎯 Atlas M4: Trading signals]
        B4[🛡️ Guardian M6: VaR, sizing, limits]
        B1 & B2 --> B3
        B3 --> B4
    end

    subgraph Phase3["Phase 3: Your decision"]
        C1[Dashboard & Command Center]
        C2[Review signals, risk, portfolio]
        C3[Approve / Reject / Modify]
        C1 --> C2 --> C3
    end

    subgraph Phase4["Phase 4: Execution & validation"]
        D1[📋 Shadow M8: Paper or live execution]
        D2[📜 Chronicle M10: Backtest if needed]
        D3[📊 Lens M11: Build views & reports]
        D1 --> D3
        D2 --> D3
    end

    Phase1 --> Phase2 --> Phase3 --> Phase4
    Phase4 -.->|Feedback| Phase1
```

---

## 3. Step-by-step: from market update to report

| Step | Phase | Who / What | Action | Output |
|------|--------|------------|--------|--------|
| **1** | Market updates | **Nexus (M1)** | Ingest prices, fundamentals, news, social | Normalized market & alternative data |
| **2** | Market updates | **Vault (M2)** | Store, validate, index | Historical + real-time datasets |
| **3** | Market updates | **Pulse (M3)** | Stream live data, compute live metrics | WebSocket feeds, live P&amp;L |
| **4** | Market updates | **Echo (M9)** | Score news/social sentiment | Sentiment signals per asset/theme |
| **5** | Analytics | **Neuron (M5)** | Run ML models (classification, prediction) | Scores, predictions, regime labels |
| **6** | Analytics | **Oracle (M7)** | Time-series & price forecasting | Price targets, scenarios |
| **7** | Signals | **Atlas (M4)** | Fuse signals, generate ideas | Trading signals, strategy suggestions |
| **8** | Risk | **Guardian (M6)** | VaR, limits, position sizing, compliance | Approved size, risk view, alerts |
| **9** | **Decision** | **👤 You** | View Dashboard / Command Center; approve, reject, or adjust | Your order / no-trade / strategy choice |
| **10** | Execution | **Shadow (M8)** | Paper or live execution, track fills | Fills, position updates |
| **11** | Validation | **Chronicle (M10)** | Backtest strategy if needed | Backtest report, metrics |
| **12** | Reporting | **Lens (M11)** | Build charts, dashboards, AI reports | Visualizations & reports |

---

## 4. User–agent interaction (sequence)

```mermaid
sequenceDiagram
    participant User as 👤 Trader
    participant UI as Dashboard / Command Center
    participant M4 as Atlas (Strategy)
    participant M6 as Guardian (Risk)
    participant M8 as Shadow (Execution)
    participant M11 as Lens (Reports)

    Note over UI,M11: Data & signals already in pipeline (M1–M7, M9)
    M4->>UI: Signals & strategy suggestions
    M6->>UI: Risk view, position sizing, limits
    UI->>User: Present options & risk
    User->>UI: Choose: approve / reject / modify
    UI->>M8: Send order (paper or live)
    M8->>UI: Fills & position update
    UI->>User: Confirmation
    User->>UI: Request report or dashboard
    UI->>M11: Build visualization / report
    M11->>UI: Charts, tables, insights
    UI->>User: Report & visualization
```

---

## 5. Where you see each agent in the UI

| Agent | Where in the platform | What you get |
|-------|------------------------|--------------|
| **Nexus (M1)** | Data Explorer, Command Center (data pipeline) | Live feeds, export, data quality |
| **Vault (M2)** | Data Explorer, exports | Historical data, validated series |
| **Pulse (M3)** | Dashboard, live charts, alerts | Real-time prices, live analytics |
| **Atlas (M4)** | Command Center → Strategies, Trading Bots | Signals, strategy ideas, bot config |
| **Neuron (M5)** | AI Models | Predictions, model outputs |
| **Guardian (M6)** | Command Center → Risk | VaR, limits, risk dashboard |
| **Oracle (M7)** | Options, predictions | Price forecasts, scenarios |
| **Shadow (M8)** | Paper trading, Portfolio (sim) | Simulated execution, paper P&amp;L |
| **Echo (M9)** | Command Center (sentiment), Social | Sentiment scores, news impact |
| **Chronicle (M10)** | Command Center → Strategies → Backtesting | Backtest results, strategy validation |
| **Lens (M11)** | Reports, Visualization, Dashboard | Charts, dashboards, AI report insights |

---

## 6. Data flow: market data → visualization

```mermaid
flowchart TB
    subgraph Inputs["External inputs"]
        I1[Market data APIs]
        I2[News & social]
        I3[Alternative data]
    end

    subgraph Layer1["Layer 1: Ingest"]
        L1A[M1 Nexus]
        L1B[M2 Vault]
        L1C[M3 Pulse]
        L1D[M9 Echo]
    end

    subgraph Layer2["Layer 2: Intelligence"]
        L2A[M5 Neuron]
        L2B[M7 Oracle]
        L2C[M4 Atlas]
        L2D[M6 Guardian]
    end

    subgraph Layer3["Layer 3: Decision & execution"]
        L3A[User decision]
        L3B[M8 Shadow]
        L3C[M10 Chronicle]
    end

    subgraph Layer4["Layer 4: Output"]
        L4A[M11 Lens]
        L4B[Dashboards]
        L4C[Reports]
        L4D[Charts]
    end

    I1 & I2 & I3 --> L1A
    L1A --> L1B & L1C & L1D
    L1B & L1C & L1D --> L2A & L2B & L2C
    L2C --> L2D
    L2D --> L3A
    L3A --> L3B & L3C
    L3B & L3C --> L4A
    L4A --> L4B & L4C & L4D
```

---

## 7. One-page reference: “From market update to report”

```
  MARKET UPDATES          ENRICHMENT              SIGNALS & RISK           YOUR DECISION           EXECUTION & REPORT
  ───────────────        ───────────             ───────────────          ─────────────           ───────────────────

  Prices, news,          Stored & validated       Fused into signals       You see:                Orders executed
  alternative data  →    (Vault M2)          →   (Atlas M4)           →   risk (Guardian M6)  →   (Shadow M8)
       │                        │                        │                        │                        │
  Nexus M1, Pulse M3,     Neuron M5,             Atlas M4,              Command Center,          Chronicle M10
  Echo M9                  Oracle M7              Guardian M6             Dashboard                 backtests
       │                        │                        │                        │                        │
       └────────────────────────┴────────────────────────┴────────────────────────┴────────────────────────┘
                                                                                              │
                                                                                              ▼
                                                                                    Lens M11 → Visualizations
                                                                                              & Reports
```

---

## Summary

- **Agents** handle: ingestion (M1, M2, M3, M9), analytics (M5, M7), signals and risk (M4, M6), execution and backtest (M8, M10), and reporting (M11).
- **You** decide: approve, reject, or modify in the Command Center and Dashboard, using agent outputs and risk views.
- **Flow**: Market updates → enrichment → signals & risk → your decision → execution → visualizations and reports (Aladdin-style decision support).

For technical orchestration details, see [AI-Agents](../wiki-content/AI-Agents.md) and [Architecture](../wiki-content/Architecture.md).
