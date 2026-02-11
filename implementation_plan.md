# Comprehensive Implementation Plan: Business Valuation AI Agent

This document provides a detailed technical roadmap for building an AI-powered financial valuation agent. The system is designed for investors and students to perform intrinsic, relative, and asset-based valuation analysis using real-time and uploaded data.

## 1. Technical Stack

| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Agent Framework** | LangGraph | Provides cyclic, stateful orchestration with built-in "interrupt" support for Human-in-the-Loop. |
| **LLMs** | Gemini, Groq, Mistral | **Multi-Provider Support**: Switch between Gemini (reasoning), Groq (speed), or Mistral (accuracy) dynamically via a factory. |
| **Frontend** | Streamlit | Efficiently handles interactive tables, sliders for HITL inputs, and financial charts. |
| **Primary Data** | yfinance | Reliable API for US-based historical financials, stock prices, and analyst estimates. |
| **Local Data (PH)** | PyPDF2 / Pandas | Tools to parse SEC Form 17-A/Q (PH) and user-uploaded valuation models in Excel. |
| **Guardrails** | Pydantic + Custom Logic | Strict schema validation for financial outputs to prevent hallucinated "magic numbers." |
| **Evaluation** | RAGAS / G-Eval | Industry standard for evaluating RAG and agentic reasoning accuracy. |
| **Deployment** | Docker | Ensures environment parity across development and production. |

## 2. System Architecture

The agent is modeled as a Stateful Directed Acyclic Graph (DAG) with explicit Proactive Suggestions, Advanced Logic Loops, and Interrupt Points.

__Agentic Workflow (Nodes):__

1. Ticker Extractor Node: Lightweight node to identify the target stock from user input.

2. Data Retrieval Node: Fetches raw financials (Income Statement, Balance Sheet, Cash Flow) for the target.

3. Analyst Planner Node (CONSOLIDATED): The "Brain" that sees both the Query and the Data. It validates intent, suggests valuation inputs (WACC, Growth, Haircuts), and selects peers in a single high-context call.

4. HITL 1: Verify Plan (Interrupt): The agent pauses. The user reviews and confirms/edits the Planner's blueprint before math occurs.

5. Financial Engine Node: A non-LLM Python tool that performs calculations: DCF, Multiples, and Liquidation models based on the approved plan.

6. Analysis & Synthesis Node: The LLM interprets the results and adds qualitative context.

7. Output Moderation Node: Final check for disclaimers and compliance.

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
│   │   ├── state.py         # State schema (TypedDict/Pydantic)
│   │   └── llm_factory.py   # Multi-provider LLM factory (NEW)
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
    Start((User Query)) --> Extractor[1. Ticker Extractor Node]
    
    %% Discovery
    Extractor --> DataFetch[2. Data Retrieval Node]
    
    %% Strategic Planning
    DataFetch --> Planner[3. Analyst Planner Node]
    Planner -- Invalid --> ErrorEnd((End: Error))
    
    %% Human-in-the-Loop
    Planner --> HITL1{{"HITL: Approve Strategy"}}
    note1[User reviews WACC, Growth, & Peers] -.-> HITL1
    
    %% Pure Calculation (Non-LLM)
    HITL1 --> Engine[4. Financial Engine Node]
    
    %% Specific Valuation Sub-processes
    subgraph "Engine Calculations"
        Engine --> DCF[DCF: FCFF & WACC]
        Engine --> Multiples[Multiples: PE, EV/EBITDA]
        Engine --> Asset[Asset: NAV & Liquidation]
    end

    %% Analysis & Moderation
    DCF --> Synthesis[5. Analysis & Synthesis Node]
    Multiples --> Synthesis
    Asset --> Synthesis
    
    Synthesis --> OutputMod[6. Output Moderation Node]
    
    %% Final Output
    OutputMod --> FinalReport((End: Final Report))

    %% Styling
    style Start fill:#333,stroke:#000,color:#fff
    style FinalReport fill:#333,stroke:#000,color:#fff
    style Planner fill:#4527A0,stroke:#311B92,color:#fff,stroke-width:2px
    style HITL1 fill:#FFB300,stroke:#E65100,color:#000,stroke-width:3px
```

## 9. Development Roadmap (Phased Approach)

* Phase 1: Foundation & Brains (COMPLETED ✅)
    * Refined `calculators.py` for WACC and Liquidation.
    * Implemented **Analyst Planner** consolidated architecture.
    * Integrated **Multi-LLM Factory** (Gemini, Groq, Mistral).
    * Enhanced Streamlit UI with assumption visibility.

* Phase 2: Intent Routing & HITL (Current Focus)
    * Implement **Non-Linear Branching**: Use the Planner's `intent` to skip the engine for simple queries.
    * Implement **LangGraph interrupts** to pause for user approval of the "Blueprint."
    * Add Source Selection for uploaded files.

* Phase 3: Visuals & Guardrails
    * Add Football Field and Sensitivity Heatmap charts.
    * Implement Pydantic-based guardrails.

* Phase 4: Expansion
    * Integrate PH Market PDF RAG.
    * Execute RAGAS-based accuracy testing.