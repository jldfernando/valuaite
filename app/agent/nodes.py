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
    """LLM suggests WACC and Growth based on company data."""
    print("\n--- [NODE] assumption_recommender_node ---")
    data = state["company_data"]
    
    try:
        rf_rate = get_risk_free_rate()
    except Exception as e:
        print(f"WARNING: Risk-free rate tool failed. Using fallback 0.045. Error: {e}")
        rf_rate = 0.045

    prompt = f"""
    Target: {state['ticker']}
    Sector: {data['market_info']['sector']}
    Industry: {data['market_info']['industry']}
    Beta: {data['market_info']['beta']}
    Risk-Free Rate: {rf_rate:.4f}
    
    Suggest the following for a DCF valuation:
    1. Weighted Average Cost of Capital (WACC) as a decimal.
    2. Terminal Growth Rate (usually 2-3%) as a decimal.
    3. 3-5 Peer Tickers for relative valuation.
    
    Format:
    WACC: [value]
    TERMINAL_GROWTH: [value]
    PEERS: [TICKER1, TICKER2, ...]
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        res_text = response.content.strip()
        print(f"LLM Response Raw: {res_text}")
        
        # Extraction logic with fallback
        wacc = float(res_text.split("WACC:")[1].strip().split("\n")[0])
        tg = float(res_text.split("TERMINAL_GROWTH:")[1].strip().split("\n")[0])
        peers = [p.strip() for p in res_text.split("PEERS:")[1].strip().split(",")]
        print(f"Success: Assumptions generated (WACC: {wacc}, TG: {tg})")
        
    except Exception as e:
        err_msg = f"LLM_ASSUMPTION_ERROR: Failed to generate/parse assumptions. {str(e)}"
        print(f"ERROR: {err_msg}. Using conservative fallbacks.")
        wacc, tg, peers = 0.09, 0.02, []
        # We don't necessarily 'fail' here, we can fallback to run the engine

    # Fetch Peer Data
    print(f"Fetching peer multiples for: {peers}")
    try:
        peer_multiples = get_peer_multiples(peers) if peers else {}
    except Exception as e:
        print(f"WARNING: Peer multiples tool failed. {e}")
        peer_multiples = {}

    return {
        "assumptions": {
            "wacc": wacc,
            "terminal_growth": tg,
            "peers": peers,
            "risk_free_rate": rf_rate
        },
        "peer_data": peer_multiples,
        "current_step": "financial_engine"
    }


def financial_engine_node(state: ValuationState) -> Dict[str, Any]:
    """Performs the actual math using calculators.py."""
    print("\n--- [NODE] financial_engine_node ---")
    data = state["company_data"]
    assump = state["assumptions"]
    
    # 1. DCF
    try:
        recent_fcf = data["cash_flow"]["operating_cash_flow"][0] + data["cash_flow"]["capital_expenditures"][0]
        print(f"Running DCF Math (FCF: {recent_fcf}, WACC: {assump['wacc']})")
        
        dcf_res = calculate_dcf(
            fcf_base=recent_fcf,
            growth_rates=0.05,
            wacc=assump["wacc"],
            terminal_growth=assump["terminal_growth"],
            shares_outstanding=data["market_info"]["shares_outstanding"],
            cash=data["balance_sheet"]["total_cash"],
            debt=data["balance_sheet"]["total_debt"]
        )
    except Exception as e:
        print(f"MATH_ERROR (DCF): {e}")
        dcf_res = {"implied_share_price": 0, "error": str(e)}

    # 2. Multiples
    try:
        target_metrics = {
            "earnings": data["income_statement"]["net_income"][0],
            "sales": data["income_statement"]["revenue"][0],
            "ebitda": data["income_statement"]["ebitda"][0],
            "net_income_growth": 0.05 
        }
        print(f"Running Multiples Math (Earnings: {target_metrics['earnings']})")
        
        multi_res = calculate_multiples_valuation(
            target_metrics=target_metrics,
            peer_multiples=state["peer_data"],
            shares_outstanding=data["market_info"]["shares_outstanding"]
        )
    except Exception as e:
        print(f"MATH_ERROR (Multiples): {e}")
        multi_res = {"error": str(e)}
    
    # 3. NAV
    try:
        print("Running NAV Math")
        nav_res = calculate_nav(
            total_assets=data["balance_sheet"]["total_assets"],
            total_liabilities=data["balance_sheet"]["total_liabilities"],
            intangible_assets=data["balance_sheet"]["intangible_assets"],
            shares_outstanding=data["market_info"]["shares_outstanding"]
        )
    except Exception as e:
        print(f"MATH_ERROR (NAV): {e}")
        nav_res = {"nav_per_share": 0, "error": str(e)}
    
    return {
        "valuation_results": {
            "dcf": dcf_res,
            "multiples": multi_res,
            "nav": nav_res
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
        
        prompt = f"""
        Subject: {ticker} Business Valuation Summary
        Current Price: ${curr_price}
        
        Intrinsic (DCF) Implied Price: ${results['dcf'].get('implied_share_price', 0):.2f}
        Relative (PE) Implied Price: ${results['multiples'].get('pe_implied_price', 0):.2f}
        Asset-Based (NAV) Per Share: ${results['nav'].get('nav_per_share', 0):.2f}
        
        Analyze these results. Is the stock potentially undervalued or overvalued? 
        Consider the margin of safety. Provide a concise financial summary for an investor.
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


