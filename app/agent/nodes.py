from datetime import datetime
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import ValuationState
from tools.finance import get_company_data, get_peer_multiples, get_risk_free_rate
from tools.calculators import (
    calculate_wacc, 
    calculate_dcf, 
    calculate_multiples_valuation, 
    calculate_nav, 
    calculate_liquidation_value
)
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM with 0 retries to prevent quota 'leaking' during rate limits
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=0
)

def input_guardrail_node(state: ValuationState) -> Dict[str, Any]:
    """Checks if the query is financial and valid."""
    print("\n--- [NODE] input_guardrail_node ---")
    
    if not os.getenv("GOOGLE_API_KEY"):
        err_msg = "AUTHENTICATION_ERROR: GOOGLE_API_KEY not found."
        print(f"ERROR: {err_msg}")
        return {"errors": [err_msg], "current_step": "error"}

    messages = state.get("messages", [])
    if not messages:
        err_msg = "INPUT_ERROR: No messages found in state."
        print(f"ERROR: {err_msg}")
        return {"errors": [err_msg], "current_step": "error"}

    last_message = messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1].content
    print(f"User Query: '{last_message}'")
    
    prompt = f"""
    Analyze the user query: "{last_message}"
    Determine if this is a request for a business valuation or financial analysis.
    If it is NOT financial, respond with 'INVALID: [Reason]'.
    If it IS financial, identify the stock ticker if present.
    Format your response as:
    STATUS: [VALID/INVALID]
    TICKER: [TICKER/NONE]
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        res_text = response.content.strip()
        
    except Exception as e:
        err_type = type(e).__name__
        err_msg = f"LLM_INVOKE_ERROR ({err_type}): {str(e)}"
        print(f"ERROR: {err_msg}")
        return {"errors": [err_msg], "current_step": "error"}
    
    if "INVALID" in res_text:
        print(f"REJECTION: {res_text}")
        return {"errors": [f"GUARDRAIL_REJECTION: {res_text}"], "current_step": "error"}
        
    # Extraction with error handling
    try:
        ticker = "UNKNOWN"
        if "TICKER:" in res_text:
            ticker = res_text.split("TICKER:")[1].strip().split("\n")[0].upper()
            if ticker == "NONE": ticker = "UNKNOWN"
        print(f"Success: Ticker '{ticker}' identified.")
    except Exception as e:
        err_msg = f"PARSING_ERROR: Could not extract ticker from response. {str(e)}"
        print(f"ERROR: {err_msg}")
        return {"errors": [err_msg], "current_step": "error"}

    return {"ticker": ticker, "current_step": "data_retrieval"}


def data_retrieval_node(state: ValuationState) -> Dict[str, Any]:
    """Fetches raw data for the target company."""
    print("\n--- [NODE] data_retrieval_node ---")
    ticker = state["ticker"]
    print(f"Executing finance tools for: {ticker}")
    
    if ticker == "UNKNOWN":
        err_msg = "VALUATION_ERROR: No valid ticker found. Cannot proceed with data retrieval."
        print(f"ERROR: {err_msg}")
        return {"errors": [err_msg]}
        
    try:
        data = get_company_data(ticker)
        if "error" in data:
            err_msg = f"FINANCE_TOOL_ERROR: {data['error']}"
            print(f"ERROR: {err_msg}")
            return {"errors": [err_msg]}
        
        print("Success: Company financials retrieved.")
        return {
            "company_data": data,
            "current_step": "assumption_recommender"
        }
    except Exception as e:
        err_msg = f"RUNTIME_ERROR (DataRetrieval): {str(e)}"
        print(f"ERROR: {err_msg}")
        return {"errors": [err_msg]}


def assumption_recommender_node(state: ValuationState) -> Dict[str, Any]:
    """LLM suggests Growth and ERP based on company data."""
    print("\n--- [NODE] assumption_recommender_node ---")
    data = state["company_data"]
    
    try:
        rf_rate = get_risk_free_rate()
    except Exception as e:
        print(f"WARNING: Risk-free rate tool failed. Using fallback 0.045. Error: {e}")
        rf_rate = 0.045

    # Prepare historical context for LLM
    hist_rev = data["income_statement"]["revenue"]
    rev_growth = [(hist_rev[i] - hist_rev[i+1])/hist_rev[i+1] if hist_rev[i+1] != 0 else 0 for i in range(len(hist_rev)-1)]
    avg_hist_growth = sum(rev_growth)/len(rev_growth) if rev_growth else 0.05

    prompt = f"""
    Target: {state['ticker']} ({data['market_info']['sector']})
    Avg Hist Revenue Growth: {avg_hist_growth:.2%}
    Recent Revenue (5Y): {hist_rev}
    Beta: {data['market_info']['beta']}

    Task: Suggest realistic valuation assumptions.
    1. 5-Year Forward Annual Revenue Growth Rate (as a decimal).
    2. Terminal Growth Rate (usually 2-3% as a decimal).
    3. Equity Risk Premium (ERP) - Standard is ~0.05-0.06.
    4. Asset Haircuts (for liquidation): Suggest a decimal for 'uncollectible' inventory and receivables.
    5. 3-5 Peer Tickers for relative valuation.

    Format:
    FORWARD_GROWTH: [value]
    TERMINAL_GROWTH: [value]
    ERP: [value]
    HAIRCUTS: [inventory_haircut, receivables_haircut]
    PEERS: [TICKER1, TICKER2, ...]
    """
    
    try:
        # Use HumanMessage for consistency
        response = llm.invoke([HumanMessage(content=prompt)])
        res_text = response.content.strip()
        print(f"LLM Response Raw: {res_text}")
        
        # Extraction logic
        fg = float(res_text.split("FORWARD_GROWTH:")[1].strip().split("\n")[0])
        tg = float(res_text.split("TERMINAL_GROWTH:")[1].strip().split("\n")[0])
        erp = float(res_text.split("ERP:")[1].strip().split("\n")[0])
        haircuts_raw = res_text.split("HAIRCUTS:")[1].strip().split("\n")[0].replace("[","").replace("]","")
        haircuts = [float(h.strip()) for h in haircuts_raw.split(",")]
        peers = [p.strip() for p in res_text.split("PEERS:")[1].strip().split(",")]
        
    except Exception as e:
        print(f"ERROR: Assumption extraction failed: {e}. Using defaults.")
        fg, tg, erp, haircuts, peers = 0.05, 0.02, 0.055, [0.5, 0.2], []

    # Fetch Peer Data
    print(f"Fetching peer multiples for: {peers}")
    peer_multiples = get_peer_multiples(peers) if peers else {}

    return {
        "assumptions": {
            "forward_growth": fg,
            "terminal_growth": tg,
            "equity_risk_premium": erp,
            "haircuts": haircuts,
            "risk_free_rate": rf_rate,
            "peers": peers
        },
        "peer_data": peer_multiples,
        "current_step": "financial_engine"
    }


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
        
        liq_res = calculate_liquidation_value(
            total_assets=data["balance_sheet"]["total_assets"],
            total_liabilities=data["balance_sheet"]["total_liabilities"],
            inventory=data["balance_sheet"]["inventory"],
            accounts_receivable=data["balance_sheet"]["accounts_receivable"],
            inventory_haircut=assump["haircuts"][0],
            receivables_haircut=assump["haircuts"][1],
            shares_outstanding=data["market_info"]["shares_outstanding"]
        )
    except Exception as e:
        print(f"MATH_ERROR (Asset-Based): {e}")
        nav_res = {"nav_per_share": 0}
        liq_res = {"liquidation_per_share": 0}
    
    return {
        "valuation_results": {
            "dcf": dcf_res,
            "multiples": multi_res,
            "nav": nav_res,
            "liquidation": liq_res,
            "calculated_wacc": calculated_wacc
        },
        "current_step": "analysis_synthesis"
    }


def analysis_synthesis_node(state: ValuationState) -> Dict[str, Any]:
    """LLM summarizes the findings."""
    print("\n--- [NODE] analysis_synthesis_node ---")
    results = state["valuation_results"]
    ticker = state["ticker"]
    
    try:
        curr_price = state["company_data"]["market_info"]["current_price"]
        wacc = results.get("calculated_wacc", 0.09)
        
        prompt = f"""
        Subject: {ticker} Professional Valuation Analysis
        Market Status: Price ${curr_price} | WACC: {wacc:.2%}
        
        Valuation Models:
        - Intrinsic (DCF) Implied Price: ${results['dcf'].get('implied_share_price', 0):.2f}
        - Relative (PE) Implied Price: ${results['multiples'].get('pe_implied_price', 0):.2f}
        - Asset-Based (NAV) Per Share: ${results['nav'].get('nav_per_share', 0):.2f}
        - Liquidation Value (Floor): ${results['liquidation'].get('liquidation_per_share', 0):.2f}
        
        Synthesis Task: 
        Analyze the variance between these models. 
        1. Compare the Current Price to the DCF and Relative valuations.
        2. Identify if the stock is trading near its 'Floor' (Liquidation Value).
        3. Conclude on the Margin of Safety.
        """
        
        print("Synthesizing final report with LLM...")
        response = llm.invoke([HumanMessage(content=prompt)])
        print("Success: Final report generated.")
        
        return {
            "analysis_report": response.content,
            "messages": [{"role": "assistant", "content": response.content}],
            "current_step": "complete"
        }
    except Exception as e:
        err_msg = f"SYNTHESIS_ERROR: Failed to generate report. {str(e)}"
        print(f"ERROR: {err_msg}")
        return {"errors": [err_msg], "current_step": "error"}


