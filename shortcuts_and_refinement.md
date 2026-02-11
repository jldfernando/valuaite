# Technical Debt & Refinement Roadmap

This document tracks "shortcuts" taken during development and provide a clear plan for refinement to ensure institutional-grade financial accuracy.

## 1. Mathematical Refinements (Phase 1 Progress)

| Item | Status | Solution Implemented |
| :--- | :--- | :--- |
| **Deterministic WACC** | ✅ **DONE** | Bypassed LLM "guessing." Now calculates WACC using `calculate_wacc()` with Risk-Free Rate (^TNX), Beta, and Cost of Debt. |
| **Contextual Growth** | ✅ **DONE** | Replaced 5% hardcoded growth. Now provides 5-year historical revenue trends to the LLM to suggest realistic forward rates. |
| **Liquidation Floor** | ✅ **DONE** | Integrated `calculate_liquidation_value()`. LLM now suggests sector-specific asset "haircuts" to find the absolute price floor. |
| **Model Stability** | ✅ **DONE** | Migrated to `gemini-2.5-flash-lite` for better availability and performance as of Feb 2026. |

## 2. Agentic & Architectural Shortcuts (Phase 2 Focus)

| Shortcut | Impact | Fix Strategy | Target Phase |
| :--- | :--- | :--- | :--- |
| **Linear Logic Flow** | Even for simple queries (e.g., "What is the P/E?"), the agent runs a full DCF. | Implement **Intent Routing** to bypass unnecessary nodes based on user query analysis. | Phase 2 |
| **No Human-in-the-Loop** | User cannot correct "hallucinated" data or adjust aggressive assumptions. | Implement **LangGraph Interrupts** to pause state and wait for user input/edits. | Phase 2 |
| **Fragile String Parsing** | Extraction of Tickers or WACC values uses `split()` and `strip()`, which is error-prone. | Use **Pydantic with Structured Output** for all LLM node responses. | Phase 3 |

## 3. Data & Robustness Shortcuts

| Shortcut | Impact | Fix Strategy | Target Phase |
| :--- | :--- | :--- | :--- |
| **Quota Leaking** | ✅ **DONE** | Set `max_retries=0` to prevent background libraries from burning daily quota on 429 errors. | Phase 1 |
| **Missing Sector Benchmarking** | "Peers" are suggested but their data isn't always contextually validated. | Implement a verification step where the LLM reviews peer business summaries before fetching multiples. | Phase 3 |
| **API Fallbacks** | If `yfinance` returns empty (NaN), the current nodes use 0.0, which can skew results. | Implement "Multi-Source" verification or explicit error messaging for missing data fields. | Phase 4 |

---

## 🚀 Suggested Next Steps

### 1. The "Analyst Planner" (Priority: High)
Consolidate the **Guardrail** and **Assumption** nodes into a single **Planning Node**. 
*   **Goal**: Reduce API calls from 3 per query to 2.
*   **Logic**: Fetch ticker data first, then ask Gemini: *"Is this valid? If so, here is the data. Give me the forward growth and peer tickers in one go."*

### 2. Human-in-the-Loop (HITL) Checkpoint
Add a pause in the graph after the Planner but before the Engine.
*   **Goal**: Let the user see the **ERP, WACC, and Peer List** in Streamlit and say "Proceed" or "Edit WACC to 10%."
*   **Tool**: Requires LangGraph `interrupts`.

### 3. Structured Outputs
Refactor the prompt logic to use Pydantic models. 
*   **Goal**: Eliminate parsing errors where the LLM returns text instead of just values.
