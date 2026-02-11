# Comprehensive Implementation Plan: Business Valuation AI Agent

This document provides a detailed technical roadmap for building an AI-powered financial valuation agent. The system is designed for investors and students to perform intrinsic, relative, and asset-based valuation analysis using real-time and uploaded data.

## 1. Technical Stack

| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Agent Framework** | LangGraph | Provides cyclic, stateful orchestration with built-in "interrupt" support for Human-in-the-Loop. |
| **LLM** | Gemini 2.0 Flash | High context window (1M+ tokens), fast, and has a generous free tier via Google AI Studio. |
| **Frontend** | Streamlit | Efficiently handles interactive tables, sliders for HITL inputs, and financial charts. |
| **Primary Data** | yfinance | Reliable API for US-based historical financials, stock prices, and analyst estimates. |
| **Local Data (PH)** | PyPDF2 / Pandas | Tools to parse SEC Form 17-A/Q (PH) and user-uploaded valuation models in Excel. |
| **Guardrails** | Pydantic + Custom Logic | Strict schema validation for financial outputs to prevent hallucinated "magic numbers." |
| **Evaluation** | RAGAS / G-Eval | Industry standard for evaluating RAG and agentic reasoning accuracy. |
| **Deployment** | Docker | Ensures environment parity across development and production. |

## 2. System Architecture

The agent is modeled as a Stateful Directed Acyclic Graph (DAG) with explicit Proactive Suggestions, Advanced Logic Loops, and Interrupt Points.

__Agentic Workflow (Nodes):__

1. Input Guardrail Node: Filters for non-financial queries and potential prompt injections.

2. Router Node: Identifies the ticker and determines the valuation method: DCF, Relative, or Asset-Based.

3. Data Retrieval Node: Fetches raw financials (Income Statement, Balance Sheet, Cash Flow) for the target and identified competitors.

4. HITL 1: Verify Raw Data (Interrupt): The agent pauses. The user reviews and confirms/edits the extracted raw data (Total Assets, Liabilities, Intangibles, etc.).

5. Assumption Recommender Node: Proposes WACC, terminal growth, peer weights, and Asset Adjustment Haircuts (e.g., discounting inventory or accounts receivable).

6. HITL 2: Approve Assumptions (Interrupt): The user reviews "Bull", "Base", or "Bear" cases. They can also request Custom Logic.

7. Advanced Logic Node: Processes technical data (Moving Averages, custom adjustments) and feeds back into the state for re-approval.

8. Financial Engine Node: A non-LLM Python tool that performs calculations: DCF formulas, Multiples benchmarks, and Net Asset Value (NAV) models.

9. Analysis & Synthesis Node: The LLM interprets the combined valuation results and adds qualitative context.

10. Output Moderation Node: Final check for disclaimers and compliance.

## 3. Financial Logic & Tooling

__Intrinsic Valuation (DCF)__

* FCFF Calculation: Derived from Net Income, D&A, Capex, and Working Capital.

* WACC: Calculated via CAPM.

__Relative Valuation (Multiples)__

The engine identifies 3-5 sector peers and calculates:

* P/E & PEG, P/B & P/S, and EV/EBITDA.

__Asset-Based Valuation (NEW)__

* Net Asset Value (NAV): (Total Assets - Intangible Assets) - Total Liabilities.

* Liquidation Value: Applies "haircuts" to current assets (e.g., 80% of Accounts Receivable, 50% of Inventory) to estimate a "worst-case" floor value.

* Book Value per Share: Direct extraction from current balance sheet data.

__Expansion for Philippine (PH) Companies__

* Strategy: Context Injection via RAG to extract balance sheet items from PH SEC Form 17-A.

## 4. Human-in-the-Loop (HITL) Implementation

* Checkback 1 (Data): Verify Balance Sheet line items for asset valuation.

* Checkback 2 (Assumptions): Allow users to adjust asset "haircuts" for liquidation scenarios.

## 5. Guardrails & Moderation

* Input: Scope checks and ticker validation.

* Output: Hallucination checks (Engine vs. LLM) and mandatory financial disclaimers.

## 6. Evaluation Plan (RAGAS)

* Faithfulness: Source data vs. Final report.

* Answer Relevance: User intent vs. Generated output.

* Financial Correctness: Comparing output against fixed Excel "Gold Standard" benchmarks.

## 7. Directory Structure
```
valuation-ai-agent/
├── app/
│   ├── main.py              # Streamlit Chat-First Entry Point
│   ├── agent/
│   │   ├── graph.py         # LangGraph state machine & logic
│   │   ├── nodes.py         # Node logic (LLM Prompts & Routing)
│   │   └── state.py         # State schema (TypedDict/Pydantic)
│   ├── tools/
│   │   ├── finance.py       # Data fetching (yfinance wrappers)
│   │   ├── calculators.py   # Math Engine (DCF, Multiples, NAV)
│   │   └── parser.py        # PDF extraction logic (Expansion)
│   └── utils/
│       └── moderation.py    # Guardrail & Disclaimer implementation
│       └── visuals.py       # Plotly/Altair chart generation
├── eval/
│   ├── gold_set.json        # Benchmark valuation targets
│   └── run_eval.py          # RAGAS evaluation script
├── Dockerfile
├── requirements.txt
└── .env
```

## 8. Flowchart
```
graph TD
    %% Starting Point
    Start((User Query)) --> Guardrail[1. Input Guardrail Node]
    
    %% Input Validation
    Guardrail -- Valid --> Router[2. Router Node]
    Guardrail -- Invalid --> ErrorEnd((End: Error))

    %% Data Acquisition
    Router --> DataFetch[3. Data Retrieval Node]
    
    %% Human-in-the-Loop 1
    DataFetch --> HITL1{{"HITL 1: Verify Raw Data"}}
    note1[User reviews extracted 10-K/yfinance data] -.-> HITL1
    
    %% Logic Recommendation
    HITL1 --> Recommender[4. Assumption Recommender Node]
    Recommender --> HITL2{{"HITL 2: Approve Assumptions"}}
    
    %% Advanced Investor Loop
    note2[User selects: Bull, Bear, or Custom Logic] -.-> HITL2
    HITL2 --> |User requests Custom Logic| AdvancedLogic[5. Advanced Logic Node]
    AdvancedLogic --> |Update Parameters| HITL2

    %% Pure Calculation (Non-LLM)
    HITL2 --> |Confirmed| Engine[6. Financial Engine Node]
    
    %% Specific Valuation Sub-processes
    subgraph "Engine Calculations"
        Engine --> DCF[DCF: FCFF & WACC]
        Engine --> Multiples[Multiples: PE, PB, PEG, PS, EV/EBITDA]
    end

    %% Analysis & Moderation
    DCF --> Synthesis[7. Analysis & Synthesis Node]
    Multiples --> Synthesis
    
    Synthesis --> OutputMod[8. Output Moderation Node]
    
    %% Final Output
    OutputMod --> FinalReport((End: Final Report))

    %% High Contrast Styling for Readability
    style Start fill:#333,stroke:#000,color:#fff
    style FinalReport fill:#333,stroke:#000,color:#fff
    
    style Guardrail fill:#C62828,stroke:#8B0000,color:#fff,stroke-width:2px
    style OutputMod fill:#C62828,stroke:#8B0000,color:#fff,stroke-width:2px
    
    style Router fill:#4527A0,stroke:#311B92,color:#fff,stroke-width:2px
    style Recommender fill:#4527A0,stroke:#311B92,color:#fff,stroke-width:2px
    style Synthesis fill:#4527A0,stroke:#311B92,color:#fff,stroke-width:2px
    
    style DataFetch fill:#2E7D32,stroke:#1B5E20,color:#fff,stroke-width:2px
    
    style AdvancedLogic fill:#1565C0,stroke:#0D47A1,color:#fff,stroke-width:2px
    style Engine fill:#1565C0,stroke:#0D47A1,color:#fff,stroke-width:2px
    
    style HITL1 fill:#FFB300,stroke:#E65100,color:#000,stroke-width:3px
    style HITL2 fill:#FFB300,stroke:#E65100,color:#000,stroke-width:3px
    
    style Engine Calculations fill:#ECEFF1,stroke:#607D8B,stroke-dasharray: 5 5,color:#000
```

## 9. Development Roadmap (Phased Approach)

* Phase 1: US Prototype (Current Priority)
    * Implement calculators.py and finance.py.
    * Build simple LangGraph flow: Data -> Engine -> Synthesis.
    * Basic Chat UI in Streamlit.

* Phase 2: Human-in-the-Loop & Interactive Logic
    * Implement LangGraph interrupts.
    * Add "Assumption Recommender" and "Advanced Logic" (Moving Averages).

* Phase 3: Visuals & Guardrails
    * Add Football Field and Sensitivity Heatmap charts.
    * Implement Pydantic-based guardrails.

* Phase 4: Expansion
    * Integrate PH Market PDF RAG.
    * Execute RAGAS-based accuracy testing.