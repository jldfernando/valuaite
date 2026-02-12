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
    provider = config.get("provider", "Groq")
    model = config.get("model", None)
    api_key = config.get("api_key", None)
    return get_llm(provider=provider, model_name=model, api_key=api_key)

def ticker_extractor_node(state: ValuationState) -> Dict[str, Any]:
    """Lightweight extraction of ticker from user query."""
    print("\n--- [NODE] ticker_extractor_node ---")
    print(f"LLM provider: {state.get('config', {}).get('provider')}")
    
    messages = state.get("messages", [])
    last_message = messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1].content
    
    prompt = f"Extract only the stock ticker from this message: '{last_message}'. If none found, respond 'NONE'."
    
    try:
        llm = get_node_llm(state)
        response = llm.invoke([HumanMessage(content=prompt)])
        ticker = response.content.strip().upper().replace("$", "").replace("TICKER:", "").strip()
        
        # If the LLM failed to find a ticker in the LAST message, 
        # check if we already have a valid ticker in the state (Follow-up case)
        if ("NONE" in ticker or len(ticker) > 10) and state.get("ticker") and state["ticker"] != "UNKNOWN":
            print(f"Follow-up detected. Retaining existing ticker: {state['ticker']}")
            ticker = state["ticker"]
        elif "NONE" in ticker or len(ticker) > 10:
            ticker = "UNKNOWN"
            
    except Exception as e:
        print(f"Ticker Extraction Failed: {e}")
        ticker = state.get("ticker", "UNKNOWN")

    print(f"Final Ticker: {ticker}")
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
    """The 'Brain' - Validates intent and sets the valuation strategy using retrieved data.
    Now conversational: reviews entire chat history to handle feedback/arguments.
    """
    print("\n--- [NODE] analyst_planner_node ---")
    data = state["company_data"]
    
    # 1. Get full conversation context
    messages = state.get("messages", [])
    chat_transcript = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        chat_transcript += f"{role.upper()}: {content}\n"

    # 2. Prepare historical context
    hist_rev = data["income_statement"]["revenue"]
    rev_growth = [(hist_rev[i] - hist_rev[i+1])/hist_rev[i+1] if hist_rev[i+1] != 0 else 0 for i in range(len(hist_rev)-1)]
    avg_hist_growth = sum(rev_growth)/len(rev_growth) if rev_growth else 0.05
    
    try:
        rf_rate = get_risk_free_rate()
    except:
        rf_rate = 0.045

    prompt = f"""
    You are a Senior Equity Research Analyst.
    
    CONVERSATION LOG:
    {chat_transcript}
    
    Company Data for {state['ticker']}:
    - Sector: {data['market_info']['sector']}
    - Beta: {data['market_info']['beta']}
    - Avg Hist Growth: {avg_hist_growth:.2%}
    - Recent Revenue: {hist_rev}

    TASK:
    1. VALIDATE: Is the user's latest message related to finance or the valuation strategy? (YES/NO)
    2. INTENT: 
       - Respond 'FULL_VALUATION' for modeling (DCF, PE, NAV) or if the user is providing FEEDBACK on your previous plan.
       - Respond 'QUICK_INQUIRY' ONLY for simple lookups.
    3. NEGOTIATE: Review the chat log. If the user challenged your previous assumptions (growth, peers, haircuts, beta, tax, etc.), either:
       - ADAPT: Change your numbers to reflect their feedback.
       - ARGUE: Briefly justify why your original number might be more accurate while still providing a middle-ground option.

    Output format (Strictly no other text):
    VALID: [YES/NO]
    INTENT: [FULL_VALUATION/QUICK_INQUIRY]
    FORWARD_GROWTH: [decimal, e.g. 0.08]
    TERMINAL_GROWTH: [decimal, e.g. 0.02]
    ERP: [decimal, e.g. 0.055]
    BETA: [decimal, e.g. 1.2]
    TAX_RATE: [decimal, e.g. 0.21]
    HAIRCUTS: [inventory_decimal, receivables_decimal]
    PEERS: [TICKER1, TICKER2, TICKER3]
    REASONING: [Conversational response addressing the user's latest point or justifying your model. Be specific about the numbers you chose.]
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
        fg_match = get_match(r"FORWARD_GROWTH:\s*([\d\.-]+)", res_text, str(avg_hist_growth))
        tg_match = get_match(r"TERMINAL_GROWTH:\s*([\d\.-]+)", res_text, "0.025")
        erp_match = get_match(r"ERP:\s*([\d\.-]+)", res_text, "0.055")
        beta_match = get_match(r"BETA:\s*([\d\.-]+)", res_text, str(data["market_info"]["beta"]))
        tax_match = get_match(r"TAX_RATE:\s*([\d\.-]+)", res_text, str(data["income_statement"]["tax_rate"]))
        haircuts_raw = get_match(r"HAIRCUTS:\s*\[?([\s\d\.,\.-]+)\]?", res_text, "0.1, 0.15")
        peers_raw = get_match(r"PEERS:\s*(.+)", res_text, "")
        reasoning = get_match(r"REASONING:\s*(.+)", res_text, "No reasoning provided.")

        # Cleanup values
        fg = float(fg_match)
        tg = float(tg_match)
        erp = float(erp_match)
        beta = float(beta_match)
        tax = float(tax_match)
        haircuts = [float(h.strip()) for h in haircuts_raw.split(",")]
        
        # Clean peers using regex for ticker patterns (1-5 uppercase chars)
        peers_extracted = re.findall(r"\b[A-Z]{1,5}\b", peers_raw)
        peers = [p for p in peers_extracted if p not in ["AND", "THE", "PEERS", "OR", state['ticker']]]

        # Logging
        print("\nPlanning results...")
        print(f"Forward Growth: {fg:.2%}")
        print(f"Terminal Growth: {tg:.2%}")
        print(f"Equity Risk Premium: {erp:.2%}")
        print(f"Beta: {beta}")
        print(f"Tax Rate: {tax:.2%}")
        print(f"Haircuts: {haircuts}")
        print(f"Peers: {peers}")
        print(f"Intent: {intent}")
        print(f"Risk Free Rate: {rf_rate:.2%}")
        print(f"Reasoning: {reasoning}")

    except Exception as e:
        print(f"Planning Error: {e}. Using conservative defaults.")
        fg, tg, erp, beta, tax, haircuts, peers, intent, reasoning = 0.05, 0.02, 0.055, 1.0, 0.25, [0.1, 0.15], [], "FULL_VALUATION", "Fallback to defaults due to planning error."

    # Fetch peers
    peer_multiples = get_peer_multiples(peers) if peers else {}
    print("\nPeer multiples...")
    print(f"Peer Multiples: {peer_multiples}")

    return sanitize_state({
        "assumptions": {
            "forward_growth": fg,
            "terminal_growth": tg,
            "equity_risk_premium": erp,
            "beta": beta,
            "tax_rate": tax,
            "haircuts": haircuts,
            "risk_free_rate": rf_rate,
            "peers": peers,
            "intent": intent,
            "reasoning": reasoning
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
            beta=assump.get("beta", data["market_info"]["beta"]),
            equity_risk_premium=assump["equity_risk_premium"],
            cost_of_debt=cost_of_debt,
            tax_rate=assump.get("tax_rate", data["income_statement"]["tax_rate"]),
            market_cap=data["market_info"]["market_cap"],
            total_debt=total_debt
        )
        print("\nWACC calculations...")
        print(f"Risk Free Rate: {assump['risk_free_rate']:.2%}")
        print(f"Beta (Negotiated): {assump.get('beta', data['market_info']['beta'])}")
        print(f"Equity Risk Premium: {assump['equity_risk_premium']:.2%}")
        print(f"Cost of Debt: {cost_of_debt:.2%}")
        print(f"Tax Rate (Negotiated): {assump.get('tax_rate', data['income_statement']['tax_rate']):.2%}")
        print(f"Market Cap: {data['market_info']['market_cap']:.2f}")
        print(f"Total Debt: {total_debt:.2f}")
        print(f"Calculated WACC: {calculated_wacc:.2%}")
    except Exception as e:
        print(f"ERROR calculating WACC: {e}. Falling back to 9%.")
        calculated_wacc = 0.09

    # 2. DCF
    try:
        # Calculate Historical FCF (usually reversed chronologically in yfinance, so reverse again)
        hist_ocf = data["cash_flow"]["operating_cash_flow"]
        hist_capex = data["cash_flow"]["capital_expenditures"]
        historical_fcf = [ocf + cape for ocf, cape in zip(hist_ocf, hist_capex)]
        
        recent_fcf = historical_fcf[0]
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
            peer_multiples=state["peer_data"].get("aggregated", {}),
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
            "calculated_wacc": calculated_wacc,
            "peer_data": state.get("peer_data", {}),
            "historical_fcf": historical_fcf
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

        # Prepare chat context for follow-up questions
        messages = state.get("messages", [])
        chat_transcript = ""
        for m in messages[:-1]: # Exclude the current prompt which is handled below
            role = m.get("role", "user")
            content = m.get("content", "")
            chat_transcript += f"{role.upper()}: {content}\n"

        if not is_full_valuation:
            prompt = f"""
            Subject: Quick Financial Inquiry for {ticker}
            Current Price: ${curr_price}
            
            CONVERSATION HISTORY:
            {chat_transcript}
            
            LATEST USER QUESTION:
            "{state['messages'][-1].get('content') if isinstance(state['messages'][-1], dict) else state['messages'][-1].content}"
            
            Using available company data (Sector: {data['market_info'].get('sector')}, Industry: {data['market_info'].get('industry')}), 
            provide a professional, direct, and CONCISE answer. 
            
            STRICT GUIDELINE: 
            If the user asked for a specific number or metric, respond ONLY with that information and a 1-sentence explanation. 
            If it is a follow-up question (e.g., "Why?"), use the conversation history to provide context.
            Do not use a standard report template or mention missing DCF models.
            """
        else:
            wacc = results.get("calculated_wacc", 0.09)
            prompt = f"""
            Subject: {ticker} Professional Valuation Analysis
            Market Status: Price ${curr_price} | WACC: {wacc:.2%}
            
            CONVERSATION LOG (FOR CONTEXT):
            {chat_transcript}

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


def scenario_analysis_node(state: ValuationState) -> Dict[str, Any]:
    """Generates Bull, Base, and Bear scenarios based on negotiated assumptions."""
    print("\n--- [NODE] scenario_analysis_node ---")
    data = state["company_data"]
    assump = state["assumptions"]
    results = state["valuation_results"]
    
    # Base DCF parameters
    recent_fcf = data["cash_flow"]["operating_cash_flow"][0] + data["cash_flow"]["capital_expenditures"][0]
    wacc = results.get("calculated_wacc", 0.09)
    shares = data["market_info"]["shares_outstanding"]
    cash = data["balance_sheet"]["total_cash"]
    debt = data["balance_sheet"]["total_debt"]
    
    # Define Scenario Multipliers
    scenarios = {
        "Bear Case": {"growth": 0.5, "terminal": 0.8, "wacc_adj": 0.01}, # 50% growth, 80% terminal, +1% WACC
        "Base Case": {"growth": 1.0, "terminal": 1.0, "wacc_adj": 0.0},
        "Bull Case": {"growth": 1.5, "terminal": 1.2, "wacc_adj": -0.01} # 150% growth, 120% terminal, -1% WACC
    }
    
    scenario_results = {}
    
    for name, adj in scenarios.items():
        try:
            res = calculate_dcf(
                fcf_base=recent_fcf,
                growth_rates=assump["forward_growth"] * adj["growth"],
                wacc=wacc + adj["wacc_adj"],
                terminal_growth=assump["terminal_growth"] * adj["terminal"],
                shares_outstanding=shares,
                cash=cash,
                debt=debt
            )
            scenario_results[name] = {
                "implied_price": res["implied_share_price"],
                "growth_used": assump["forward_growth"] * adj["growth"],
                "terminal_used": assump["terminal_growth"] * adj["terminal"],
                "wacc_used": wacc + adj["wacc_adj"],
                "assumptions_desc": f"Growth: {adj['growth']}x | Terminal: {adj['terminal']}x | WACC: {adj['wacc_adj']:+.0%}"
            }
        except Exception as e:
            print(f"Error calculating {name}: {e}")
            
    return sanitize_state({
        "scenarios": scenario_results,
        "current_step": "complete"
    })


