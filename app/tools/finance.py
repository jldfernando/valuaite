import yfinance as yf
import pandas as pd
from typing import Dict, List, Any, Optional

def get_risk_free_rate() -> float:
    """
    Fetches the current yield of the 10-year US Treasury (^TNX).
    
    Returns:
        Risk-free rate as a decimal (e.g., 0.042 for 4.2%).
    """
    try:
        tnx = yf.Ticker("^TNX")
        # Get the latest close price
        hist = tnx.history(period="1d")
        if not hist.empty:
            # ^TNX price is the percentage (e.g., 4.2), so we divide by 100
            return float(hist['Close'].iloc[-1]) / 100
        return 0.04 # Default fallback
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return 0.04

def get_peers(ticker_symbol: str) -> Dict[str, str]:
    """
    Returns the sector and industry of a company. 
    The LLM will use this info to suggest peers.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        return {
            "sector": ticker.info.get("sector", "Unknown"),
            "industry": ticker.info.get("industry", "Unknown"),
            "business_summary": ticker.info.get("longBusinessSummary", "")
        }
    except Exception as e:
        return {"error": str(e)}

def get_peer_multiples(tickers: List[str]) -> Dict[str, List[float]]:
    """
    Fetches valuation multiples for a list of peer tickers.
    
    Args:
        tickers: List of tickers suggested by the LLM (e.g., ['MSFT', 'GOOGL']).
        
    Returns:
        Dictionary of lists: {'pe': [...], 'ps': [...], 'ev_ebitda': [...], 'peg': [...]}
    """
    multiples = {"pe": [], "ps": [], "ev_ebitda": [], "peg": []}
    
    for t_sym in tickers:
        try:
            t = yf.Ticker(t_sym)
            info = t.info
            
            # Safely grab multiples, skipping if missng
            if info.get("forwardPE"): multiples["pe"].append(float(info["forwardPE"]))
            if info.get("priceToSalesTrailing12Months"): multiples["ps"].append(float(info["priceToSalesTrailing12Months"]))
            if info.get("enterpriseToEbitda"): multiples["ev_ebitda"].append(float(info["enterpriseToEbitda"]))
            if info.get("pegRatio"): multiples["peg"].append(float(info["pegRatio"]))
            
        except Exception as e:
            print(f"Skipping {t_sym} due to error: {e}")
            continue
            
    return multiples


def get_company_data(ticker_symbol: str) -> Dict[str, Any]:
    """
    Fetches comprehensive financial data for a given ticker.
    
    Args:
        ticker_symbol: The stock ticker (e.g., 'AAPL').
        
    Returns:
        A dictionary containing cleaned financial statements and market info.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # 1. Market Info
        market_info = {
            "current_price": info.get("currentPrice", 0),
            "market_cap": info.get("marketCap", 0),
            "beta": info.get("beta", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "currency": info.get("currency", "USD")
        }

        # 2. Financial Statements (DataFrames)
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cash_flow

        def clean_df(df) -> Dict:
            if df is None or df.empty:
                return {}
            # Convert NaN to 0 and get dictionary with string keys (dates)
            return df.fillna(0).to_dict()

        # Helper to safely get a value from a DF or default to 0
        def safe_get_annual(df, key, years=5):
            if df is None or df.empty or key not in df.index:
                return [0.0] * years
            # Return list of values for the last N years
            vals = df.loc[key].tolist()
            # Clean values: replace nan with 0 and ensure float
            import math
            cleaned_vals = [float(v) if (v is not None and not (isinstance(v, float) and math.isnan(v))) else 0.0 for v in vals]
            # Pad with 0.0s if less than requested years
            return (cleaned_vals + [0.0] * years)[:years]

        # 3. Income Statement Data
        income_stmt_data = {
            "revenue": safe_get_annual(financials, "Total Revenue"),
            "ebitda": safe_get_annual(financials, "EBITDA"),
            "net_income": safe_get_annual(financials, "Net Income"),
            "interest_expense": safe_get_annual(financials, "Interest Expense"),
            "tax_rate": info.get("effectiveTaxRate", 0.25) # Use info or default
        }

        # 4. Balance Sheet Data
        # Balance sheet usually just needs the most recent for NAV/Liquidation
        bs_data = {
            "total_cash": info.get("totalCash", 0),
            "total_debt": info.get("totalDebt", 0),
            "total_assets": balance_sheet.loc["Total Assets"].iloc[0] if "Total Assets" in balance_sheet.index else 0,
            "total_liabilities": balance_sheet.loc["Total Liabilities Net Minority Interest"].iloc[0] if "Total Liabilities Net Minority Interest" in balance_sheet.index else 0,
            "intangible_assets": balance_sheet.loc["Intangible Assets"].iloc[0] if "Intangible Assets" in balance_sheet.index else 0,
            "inventory": balance_sheet.loc["Inventory"].iloc[0] if "Inventory" in balance_sheet.index else 0,
            "accounts_receivable": balance_sheet.loc["Receivables"].iloc[0] if "Receivables" in balance_sheet.index else 0,
        }

        # 5. Cash Flow Data
        cf_data = {
            "operating_cash_flow": safe_get_annual(cash_flow, "Operating Cash Flow"),
            "capital_expenditures": safe_get_annual(cash_flow, "Capital Expenditure")
        }

        return {
            "ticker": ticker_symbol,
            "market_info": market_info,
            "income_statement": income_stmt_data,
            "balance_sheet": bs_data,
            "cash_flow": cf_data,
            "raw_financials": clean_df(financials), # Including raw for flexibility
            "raw_balance_sheet": clean_df(balance_sheet),
            "raw_cash_flow": clean_df(cash_flow)
        }

    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return {"error": str(e), "ticker": ticker_symbol}
