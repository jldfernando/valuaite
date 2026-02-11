# Technical Debt & Refinement Roadmap

This document tracks "shortcuts" taken during the initial Phase 1 prototype development and provides a clear plan for refinement to ensure institutional-grade financial accuracy.

## 1. Mathematical Shortcuts

| Shortcut | Impact | Fix Strategy | Target Phase |
| :--- | :--- | :--- | :--- |
| **Bypassed `calculate_wacc`** | Valuation is based on LLM "estimates" rather than balance sheet math. | Update `financial_engine_node` to use `calculate_wacc()` tool with Beta and Market Cap data. | Phase 1 (Immediate) |
| **Hardcoded Growth Rates** | DCF uses a flat 5% growth for all companies regardless of profile. | Extract 5yr historical revenue growth in `data_retrieval` and have LLM suggest a context-aware rate. | Phase 2 |
| **Missing Liquidation Floor** | The "Asset-Based" valuation is currently limited to Book Value (NAV) only. | Integrate `calculate_liquidation_value()` with asset "haircuts" into the Engine node. | Phase 2 / 3 |

## 2. Agentic & Architectural Shortcuts

| Shortcut | Impact | Fix Strategy | Target Phase |
| :--- | :--- | :--- | :--- |
| **Linear Logic Flow** | Even for simple queries (e.g., "What is the P/E?"), the agent runs a full DCF. | Implement **Intent Routing** to bypass unnecessary nodes based on user query analysis. | Phase 2 |
| **No Human-in-the-Loop** | User cannot correct "hallucinated" data or adjust aggressive assumptions. | Implement **LangGraph Interrupts** to pause state and wait for user input/edits. | Phase 2 |
| **Fragile String Parsing** | Extraction of Tickers or WACC values uses `split()` and `strip()`, which is error-prone. | Use **Pydantic with Structured Output** (`with_structured_output`) for all LLM node responses. | Phase 3 |

## 3. Data & Robustness Shortcuts

| Shortcut | Impact | Fix Strategy | Target Phase |
| :--- | :--- | :--- | :--- |
| **Empty requirements.txt** | Dependencies were manually installed but not fully pinned. | Perform a full `pip freeze` and cleanup of `requirements.txt`. | Phase 1 (Done) |
| **Missing Sector Benchmarking** | "Peers" are suggested but their data isn't always contextually validated. | Implement a verification step where the LLM reviews peer business summaries before fetching multiples. | Phase 3 |
| **API Fallbacks** | If `yfinance` returns empty (NaN), the current nodes use 0.0, which can skew results. | Implement "Multi-Source" verification or explicit error messaging for missing data fields. | Phase 4 |

---

## Next Steps Priorities
1. **Refactor `nodes.py`**: Transition from LLM "guessing" to deterministic math using our existing tools.
2. **Setup Graph Interrupts**: Allow for the first HITL checkpoint.
3. **Structured Response Objects**: Move away from raw string manipulation for safer state transitions.
