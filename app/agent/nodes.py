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
    model="gemini-2.0-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=0
)

def input_guardrail_node(state: ValuationState) -> Dict[str, Any]:
    """Checks if the query is financial and valid."""
    print("--- [NODE] input_guardrail_node ---")
    
    # Secure API Key Check
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found in environment.")
        return {"errors": ["API Key missing. Please set GOOGLE_API_KEY in your .env file."], "current_step": "error"}

    messages = state.get("messages", [])
    if not messages:
        print("ERROR: No messages found in state.")
        return {"errors": ["Empty user message."], "current_step": "error"}

    last_message = messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1].content
    print(f"User Message: {last_message}")
    
    prompt = f"""
    Analyze the user query: "{last_message}"
    Determine if this is a request for a business valuation or financial analysis.
    If it is NOT financial, respond with 'INVALID: [Reason]'.
    If it IS financial, identify the stock ticker if present.
    Format your response as:
    STATUS: [VALID/INVALID]
    TICKER: [TICKER/NONE]
    """
    
    # Use HumanMessage as it's more universally supported across Gemini API versions
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        res_text = response.content
        print(f"Guardrail Response: {res_text.strip()}")
    except Exception as e:
        print(f"LLM Invoke Error: {e}")
        return {"errors": [f"LLM Error: {str(e)}"], "current_step": "error"}
    
    if "INVALID" in res_text:
        return {"errors": [res_text], "current_step": "error"}
        
    # Extract ticker
    ticker = "UNKNOWN"
    if "TICKER:" in res_text:
        ticker = res_text.split("TICKER:")[1].strip().split("\n")[0].upper()
        if ticker == "NONE": ticker = "UNKNOWN"
    
    print(f"Extracted Ticker: {ticker}")
    return {"ticker": ticker, "current_step": "data_retrieval"}


def data_retrieval_node(state: ValuationState) -> Dict[str, Any]:
    """Fetches raw data for the target company."""
    print("--- [NODE] data_retrieval_node ---")
    ticker = state["ticker"]
    print(f"Fetching data for: {ticker}")
    if ticker == "UNKNOWN":
        return {"errors": ["No ticker identified. Please provide a stock symbol (e.g., AAPL)."]}
        
    data = get_company_data(ticker)
    if "error" in data:
        print(f"Error fetching data: {data['error']}")
        return {"errors": [data["error"]]}
        
    print("Successfully fetched company data.")
    return {
        "company_data": data,
        "current_step": "assumption_recommender"
    }

def assumption_recommender_node(state: ValuationState) -> Dict[str, Any]:
    """LLM suggests WACC and Growth based on company data."""
    print("--- [NODE] assumption_recommender_node ---")
    data = state["company_data"]
    rf_rate = get_risk_free_rate()
    print(f"Current Risk-Free Rate: {rf_rate}")
    
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
    
    # Use HumanMessage for consistency
    response = llm.invoke([HumanMessage(content=prompt)])
    res_text = response.content
    print(f"LLM Assumptions: {res_text.strip()}")
    
    # Simple extraction logic
    try:
        wacc = float(res_text.split("WACC:")[1].strip().split("\n")[0])
        tg = float(res_text.split("TERMINAL_GROWTH:")[1].strip().split("\n")[0])
        peers = [p.strip() for p in res_text.split("PEERS:")[1].strip().split(",")]
    except Exception as e:
        print(f"Error parsing assumptions: {e}")
        # Fallbacks
        wacc, tg, peers = 0.09, 0.02, []

    print(f"Parsed -> WACC: {wacc}, TG: {tg}, Peers: {peers}")
    # Fetch Peer Data
    peer_multiples = get_peer_multiples(peers) if peers else {}
    print(f"Fetched Multiples for {len(peers)} peers.")

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
    print("--- [NODE] financial_engine_node ---")
    data = state["company_data"]
    assump = state["assumptions"]
    
    # 1. DCF
    # Use the most recent FCF (Operating - Capex)
    try:
        recent_fcf = data["cash_flow"]["operating_cash_flow"][0] + data["cash_flow"]["capital_expenditures"][0]
        print(f"Calculating DCF with Base FCF: {recent_fcf}")
        
        dcf_res = calculate_dcf(
            fcf_base=recent_fcf,
            growth_rates=0.05, # Simple 5% growth fallback for protype
            wacc=assump["wacc"],
            terminal_growth=assump["terminal_growth"],
            shares_outstanding=data["market_info"]["shares_outstanding"],
            cash=data["balance_sheet"]["total_cash"],
            debt=data["balance_sheet"]["total_debt"]
        )
    except Exception as e:
        print(f"DCF Calculation Error: {e}")
        dcf_res = {"implied_share_price": 0}

    # 2. Multiples
    # Prepare metrics
    try:
        target_metrics = {
            "earnings": data["income_statement"]["net_income"][0],
            "sales": data["income_statement"]["revenue"][0],
            "ebitda": data["income_statement"]["ebitda"][0],
            "net_income_growth": 0.05 
        }
        print(f"Calculating Multiples for Target Metrics: {target_metrics}")
        
        multi_res = calculate_multiples_valuation(
            target_metrics=target_metrics,
            peer_multiples=state["peer_data"],
            shares_outstanding=data["market_info"]["shares_outstanding"]
        )
    except Exception as e:
        print(f"Multiples Calculation Error: {e}")
        multi_res = {}
    
    # 3. NAV
    try:
        print("Calculating NAV...")
        nav_res = calculate_nav(
            total_assets=data["balance_sheet"]["total_assets"],
            total_liabilities=data["balance_sheet"]["total_liabilities"],
            intangible_assets=data["balance_sheet"]["intangible_assets"],
            shares_outstanding=data["market_info"]["shares_outstanding"]
        )
    except Exception as e:
        print(f"NAV Calculation Error: {e}")
        nav_res = {"nav_per_share": 0}
    
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
    print("--- [NODE] analysis_synthesis_node ---")
    results = state["valuation_results"]
    ticker = state["ticker"]
    curr_price = state["company_data"]["market_info"]["current_price"]
    
    prompt = f"""
    Subject: {ticker} Business Valuation Summary
    Current Price: ${curr_price}
    
    Intrinsic (DCF) Implied Price: ${results['dcf']['implied_share_price']:.2f}
    Relative (PE) Implied Price: ${results['multiples'].get('pe_implied_price', 0):.2f}
    Asset-Based (NAV) Per Share: ${results['nav']['nav_per_share']:.2f}
    
    Analyze these results. Is the stock potentially undervalued or overvalued? 
    Consider the margin of safety. Provide a concise financial summary for an investor.
    """
    
    print("Generating final summary with LLM...")
    response = llm.invoke([HumanMessage(content=prompt)])
    print("Final report generated.")
    
    return {
        "analysis_report": response.content,
        "messages": [{"role": "assistant", "content": response.content}],
        "current_step": "complete"
    }

