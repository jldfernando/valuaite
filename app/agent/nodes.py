from datetime import datetime
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from agent.state import ValuationState
from agent.llm_factory import get_llm
from tools.finance import get_company_data, get_peer_multiples, get_risk_free_rate
from tools.calculators import (
    calculate_wacc, 
    calculate_dcf, 
    calculate_multiples_valuation, 
    calculate_nav, 
    calculate_liquidation_value
)
from agent.utils import sanitize_state

load_dotenv()

def get_node_llm(state: ValuationState):
    """Helper to get the configured LLM for the current node."""
    config = state.get("config", {})
    provider = config.get("provider", "Gemini")
    model = config.get("model", None)
    api_key = config.get("api_key", None)
    return get_llm(provider=provider, model_name=model, api_key=api_key)

def ticker_extractor_node(state: ValuationState) -> Dict[str, Any]:
    """Lightweight extraction of ticker from user query."""
    print("\n--- [NODE] ticker_extractor_node ---")
    
    messages = state.get("messages", [])
    last_message = messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1].content
    
    prompt = f"Extract only the stock ticker from this message: '{last_message}'. If none found, respond 'NONE'."
    
    try:
        llm = get_node_llm(state)
        response = llm.invoke([HumanMessage(content=prompt)])
        ticker = response.content.strip().upper().replace("$", "").replace("TICKER:", "").strip()
        if "NONE" in ticker or len(ticker) > 10: 
            ticker = "UNKNOWN"
    except Exception as e:
        print(f"Ticker Extraction Failed: {e}")
        ticker = "UNKNOWN"

    print(f"Extracted Ticker: {ticker}")
    return sanitize_state({
        "ticker": ticker, 
        "current_step": "data_retrieval"
    })


def data_retrieval_node(state: ValuationState) -> Dict[str, Any]:
    """Fetches raw data for the target company."""
    print("\n--- [NODE] data_retrieval_node ---")
    ticker = state["ticker"]
    
    if ticker == "UNKNOWN":
        err_msg = "VALUATION_ERROR: I couldn't identify a stock symbol. Please mention a ticker like 'AAPL' or 'TSLA'."
        print(f"ERROR: {err_msg}")
        return sanitize_state({"errors": [err_msg], "current_step": "error"})
        
    try:
        data = get_company_data(ticker)
        if "error" in data:
            return sanitize_state({"errors": [f"DATA_ERROR: {data['error']}"], "current_step": "error"})
        
        return sanitize_state({
            "company_data": data,
            "current_step": "analyst_planner"
        })
    except Exception as e:
        return sanitize_state({"errors": [f"RUNTIME_ERROR: {str(e)}"], "current_step": "error"})


def analyst_planner_node(state: ValuationState) -> Dict[str, Any]:
    """The 'Brain' - Validates intent and sets the valuation strategy using retrieved data."""
    print("\n--- [NODE] analyst_planner_node ---")
    data = state["company_data"]
    user_query = (state["messages"][-1]["content"] if isinstance(state["messages"][-1], dict) 
                  else state["messages"][-1].content)
    
    # 1. Prepare historical context
    hist_rev = data["income_statement"]["revenue"]
    rev_growth = [(hist_rev[i] - hist_rev[i+1])/hist_rev[i+1] if hist_rev[i+1] != 0 else 0 for i in range(len(hist_rev)-1)]
    avg_hist_growth = sum(rev_growth)/len(rev_growth) if rev_growth else 0.05
    
    try:
        rf_rate = get_risk_free_rate()
    except:
        rf_rate = 0.045

    prompt = f"""
    You are a Senior Equity Research Analyst.
    User Question: "{user_query}"
    
    Company Data for {state['ticker']}:
    - Sector: {data['market_info']['sector']}
    - Beta: {data['market_info']['beta']}
    - Avg Hist Growth: {avg_hist_growth:.2%}
    - Recent Revenue: {hist_rev}

    TASK:
    1. VALIDATE: Is this question related to finance, stock market, or company analysis? (YES/NO)
    2. INTENT: 
       - Respond 'FULL_VALUATION' if the user asks for "value", "fair price", "intrinsic worth", "DCF", "PE valuation", "NAV", "liquidation", or a full report.
       - Respond 'QUICK_INQUIRY' ONLY for simple data lookups (e.g., "What is the beta?", "Who are competitors?", "What sector is this?").
       - If in doubt, default to 'FULL_VALUATION'.
    3. STRATEGY: Suggest DCF growth, Equity Risk Premium (standard ~0.055), asset haircuts, and 4-6 peer tickers.

    Output format (Strictly no other text):
    VALID: [YES/NO]
    INTENT: [FULL_VALUATION/QUICK_INQUIRY]
    FORWARD_GROWTH: [decimal, e.g. 0.08]
    ERP: [decimal, e.g. 0.055]
    HAIRCUTS: [inventory_decimal, receivables_decimal]
    PEERS: [TICKER1, TICKER2, TICKER3]
    REASONING: [Brief explanation]
    """
    
    try:
        llm = get_node_llm(state)
        response = llm.invoke([HumanMessage(content=prompt)])
        res_text = response.content.strip()
        print(f"Planner Response:\n{res_text}")

        # Guardrail logic
        if "VALID: NO" in res_text:
            return sanitize_state({"errors": ["I am specialized in financial valuations. Please ask a finance-related question."], "current_step": "error"})

        # More robust extraction using regex
        import re
        def get_match(pattern, text, default=None):
            match = re.search(pattern, text)
            return match.group(1).strip() if match else default

        intent = get_match(r"INTENT:\s*(FULL_VALUATION|QUICK_INQUIRY)", res_text, "FULL_VALUATION")
        fg_match = get_match(r"FORWARD_GROWTH:\s*([\d\.-]+)", res_text, "0.05")
        erp_match = get_match(r"ERP:\s*([\d\.-]+)", res_text, "0.055")
        haircuts_raw = get_match(r"HAIRCUTS:\s*\[?([\s\d\.,\.-]+)\]?", res_text, "0.1, 0.15")
        peers_raw = get_match(r"PEERS:\s*(.+)", res_text, "")
        reasoning = get_match(r"REASONING:\s*(.+)", res_text, "No reasoning provided.")

        # Cleanup values
        fg = float(fg_match)
        erp = float(erp_match)
        haircuts = [float(h.strip()) for h in haircuts_raw.split(",")]
        
        # Clean peers using regex for ticker patterns (1-5 uppercase chars)
        peers_extracted = re.findall(r"\b[A-Z]{1,5}\b", peers_raw)
        peers = [p for p in peers_extracted if p not in ["AND", "THE", "PEERS", "OR", state['ticker']]]

        # Logging
        print("\nPlanning results...")
        print(f"Forward Growth: {fg:.2%}")
        print(f"Equity Risk Premium: {erp:.2%}")
        print(f"Haircuts: {haircuts}")
        print(f"Peers: {peers}")
        print(f"Intent: {intent}")
        print(f"Risk Free Rate: {rf_rate:.2%}")
        print(f"Reasoning: {reasoning}")

    except Exception as e:
        print(f"Planning Error: {e}. Using conservative defaults.")
        fg, erp, haircuts, peers, intent, reasoning = 0.05, 0.055, [0.5, 0.2], [], "FULL_VALUATION", ""

    # Fetch peers
    peer_multiples = get_peer_multiples(peers) if peers else {}
    print("\nPeer multiples...")
    print(f"Peer Multiples: {peer_multiples}")

    return sanitize_state({
        "assumptions": {
            "forward_growth": fg,
            "terminal_growth": 0.025,
            "equity_risk_premium": erp,
            "haircuts": haircuts,
            "risk_free_rate": rf_rate,
            "peers": peers,
            "intent": intent
        },
        "peer_data": peer_multiples,
        "current_step": "financial_engine"
    })


def financial_engine_node(state: ValuationState) -> Dict[str, Any]:
    """Performs the actual math using calculators.py."""
    print("\n--- [NODE] financial_engine_node ---")
    data = state["company_data"]
    assump = state["assumptions"]
    
    # 1. Deterministic WACC Calculation
    try:
        # Calculate Cost of Debt: Interest Expense / Total Debt
        total_debt = data["balance_sheet"]["total_debt"]
        interest_exp = data["income_statement"]["interest_expense"][0]
        cost_of_debt = abs(interest_exp / total_debt) if total_debt > 0 else 0.05
        
        calculated_wacc = calculate_wacc(
            risk_free_rate=assump["risk_free_rate"],
            beta=data["market_info"]["beta"],
            equity_risk_premium=assump["equity_risk_premium"],
            cost_of_debt=cost_of_debt,
            tax_rate=data["income_statement"]["tax_rate"],
            market_cap=data["market_info"]["market_cap"],
            total_debt=total_debt
        )
        print("\nWACC calculations...")
        print(f"Risk Free Rate: {assump['risk_free_rate']:.2%}")
        print(f"Beta: {data['market_info']['beta']:.2%}")
        print(f"Equity Risk Premium: {assump['equity_risk_premium']:.2%}")
        print(f"Cost of Debt: {cost_of_debt:.2%}")
        print(f"Tax Rate: {data['income_statement']['tax_rate']:.2%}")
        print(f"Market Cap: {data['market_info']['market_cap']:.2%}")
        print(f"Total Debt: {total_debt:.2%}")
        print(f"Calculated WACC: {calculated_wacc:.2%}")
    except Exception as e:
        print(f"ERROR calculating WACC: {e}. Falling back to 9%.")
        calculated_wacc = 0.09

    # 2. DCF
    try:
        recent_fcf = data["cash_flow"]["operating_cash_flow"][0] + data["cash_flow"]["capital_expenditures"][0]
        print(f"Running DCF Math (Growth: {assump['forward_growth']:.2%})")
        
        dcf_res = calculate_dcf(
            fcf_base=recent_fcf,
            growth_rates=assump["forward_growth"],
            wacc=calculated_wacc,
            terminal_growth=assump["terminal_growth"],
            shares_outstanding=data["market_info"]["shares_outstanding"],
            cash=data["balance_sheet"]["total_cash"],
            debt=total_debt
        )
        print("\nDCF calculations...")
        print(f"FCF Base: {recent_fcf:.2f}")
        print(f"Growth Rates: {assump['forward_growth']:.2%}")
        print(f"WACC: {calculated_wacc:.2%}")
        print(f"Terminal Growth: {assump['terminal_growth']:.2%}")
        print(f"Shares Outstanding: {data['market_info']['shares_outstanding']:.2f}")
        print(f"Cash: {data['balance_sheet']['total_cash']:.2f}")
        print(f"Debt: {total_debt:.2f}")
        print(f"Implied Share Price: {dcf_res['implied_share_price']:.2f}")
    except Exception as e:
        print(f"MATH_ERROR (DCF): {e}")
        dcf_res = {"implied_share_price": 0, "error": str(e)}

    # 3. Multiples
    try:
        target_metrics = {
            "earnings": data["income_statement"]["net_income"][0],
            "sales": data["income_statement"]["revenue"][0],
            "ebitda": data["income_statement"]["ebitda"][0],
            "net_income_growth": assump["forward_growth"] 
        }
        multi_res = calculate_multiples_valuation(
            target_metrics=target_metrics,
            peer_multiples=state["peer_data"],
            shares_outstanding=data["market_info"]["shares_outstanding"]
        )
        print("\nMultiples calculations...")
        print(f"Target Metrics: {target_metrics}")
        print(f"Peer Multiples: {state['peer_data']}")
        print(f"Shares Outstanding: {data['market_info']['shares_outstanding']:.2f}")
        implied_pe_price = multi_res.get('pe_implied_price', 0)
        print(f"Implied Share Price (PE): {implied_pe_price:.2f}")
    except Exception as e:
        print(f"MATH_ERROR (Multiples): {e}")
        multi_res = {"error": str(e)}
    
    # 4. Asset-Based (NAV & Liquidation)
    try:
        nav_res = calculate_nav(
            total_assets=data["balance_sheet"]["total_assets"],
            total_liabilities=data["balance_sheet"]["total_liabilities"],
            intangible_assets=data["balance_sheet"]["intangible_assets"],
            shares_outstanding=data["market_info"]["shares_outstanding"]
        )
        print("\nAsset-Based calculations...")
        print(f"Total Assets: {data['balance_sheet']['total_assets']:.2f}")
        print(f"Total Liabilities: {data['balance_sheet']['total_liabilities']:.2f}")
        print(f"Intangible Assets: {data['balance_sheet']['intangible_assets']:.2f}")
        print(f"Shares Outstanding: {data['market_info']['shares_outstanding']:.2f}")
        nav_price = nav_res.get('nav_per_share', 0)
        print(f"Implied Share Price (NAV): {nav_price:.2f}")
        
        liq_res = calculate_liquidation_value(
            total_assets=data["balance_sheet"]["total_assets"],
            total_liabilities=data["balance_sheet"]["total_liabilities"],
            inventory=data["balance_sheet"]["inventory"],
            accounts_receivable=data["balance_sheet"]["accounts_receivable"],
            inventory_haircut=assump["haircuts"][0],
            receivables_haircut=assump["haircuts"][1],
            shares_outstanding=data["market_info"]["shares_outstanding"]
        )
        print("\nLiquidation calculations...")
        print(f"Total Assets: {data['balance_sheet']['total_assets']:.2f}")
        print(f"Total Liabilities: {data['balance_sheet']['total_liabilities']:.2f}")
        print(f"Inventory: {data['balance_sheet']['inventory']:.2f}")
        print(f"Accounts Receivable: {data['balance_sheet']['accounts_receivable']:.2f}")
        print(f"Inventory Haircut: {assump['haircuts'][0]:.2%}")
        print(f"Receivables Haircut: {assump['haircuts'][1]:.2%}")
        print(f"Shares Outstanding: {data['market_info']['shares_outstanding']:.2f}")
        print(f"Implied Share Price: {liq_res['liquidation_per_share']:.2f}")
    except Exception as e:
        print(f"MATH_ERROR (Asset-Based): {e}")
        nav_res = {"nav_per_share": 0}
        liq_res = {"liquidation_per_share": 0}
    
    return sanitize_state({
        "valuation_results": {
            "dcf": dcf_res,
            "multiples": multi_res,
            "nav": nav_res,
            "liquidation": liq_res,
            "calculated_wacc": calculated_wacc
        },
        "current_step": "analysis_synthesis"
    })


def analysis_synthesis_node(state: ValuationState) -> Dict[str, Any]:
    """LLM summarizes the findings, handling both full valuations and quick inquiries."""
    print("\n--- [NODE] analysis_synthesis_node ---")
    results = state.get("valuation_results", {})
    ticker = state["ticker"]
    assump = state.get("assumptions", {})
    intent = assump.get("intent", "FULL_VALUATION")
    
    try:
        data = state["company_data"]
        curr_price = data["market_info"]["current_price"]
        
        # FAIL-SAFE: If we have dcf results but intent was QUICK_INQUIRY, treat as FULL_VALUATION
        # This handles cases where the router overrode the intent or the planner was inconsistent.
        is_full_valuation = (intent == "FULL_VALUATION") or ("dcf" in results and results["dcf"].get("implied_share_price", 0) > 0)

        if not is_full_valuation:
            prompt = f"""
            Subject: Quick Financial Inquiry for {ticker}
            Current Price: ${curr_price}
            
            Task:
            The user has a specific question about {ticker}: "{state['messages'][-1].content if hasattr(state['messages'][-1], 'content') else state['messages'][-1]['content']}"
            
            Using the available company data (Sector: {data['market_info']['sector']}, Industry: {data['market_info']['industry']}), 
            provide a professional and concise answer. Do not mention missing DCF models unless strictly necessary.
            """
        else:
            wacc = results.get("calculated_wacc", 0.09)
            prompt = f"""
            Subject: {ticker} Professional Valuation Analysis
            Market Status: Price ${curr_price} | WACC: {wacc:.2%}
            
            Valuation Models:
            - Intrinsic (DCF) Implied Price: ${results.get('dcf', {}).get('implied_share_price', 0):.2f}
            - Relative (PE) Implied Price: ${results.get('multiples', {}).get('pe_implied_price', 0):.2f}
            - Asset-Based (NAV) Per Share: ${results.get('nav', {}).get('nav_per_share', 0):.2f}
            - Liquidation Value (Floor): ${results.get('liquidation', {}).get('liquidation_per_share', 0):.2f}
            
            Synthesis Task: 
            Analyze the variance between these models. 
            1. Compare the Current Price to the DCF and Relative valuations.
            2. Identify if the stock is trading near its 'Floor' (Liquidation Value).
            3. Conclude on the Margin of Safety.
            """
        
        print("Synthesizing final report with LLM...")
        llm = get_node_llm(state)
        response = llm.invoke([HumanMessage(content=prompt)])
        print("Success: Final report generated.")
        
        return sanitize_state({
            "analysis_report": response.content,
            "messages": [{"role": "assistant", "content": response.content}],
            "current_step": "complete"
        })
    except Exception as e:
        err_msg = f"SYNTHESIS_ERROR: Failed to generate report. {str(e)}"
        print(f"ERROR: {err_msg}")
        return sanitize_state({"errors": [err_msg], "current_step": "error"})


