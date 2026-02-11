from typing import Dict, List, Union, Optional
import numpy as np

def calculate_wacc(
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float,
    cost_of_debt: float,
    tax_rate: float,
    market_cap: float,
    total_debt: float
) -> float:
    """
    Calculates the Weighted Average Cost of Capital (WACC).
    
    Args:
        risk_free_rate: Usually 10-year Treasury yield.
        beta: Stock's sensitivity to market moves.
        equity_risk_premium: Average market return above risk-free rate.
        cost_of_debt: Average interest rate on debt.
        tax_rate: Corporate tax rate (decimal).
        market_cap: Total market value of equity.
        total_debt: Total debt on balance sheet.
        
    Returns:
        WACC as a decimal.
    """
    cost_of_equity = risk_free_rate + (beta * equity_risk_premium)
    
    total_value = market_cap + total_debt
    
    if total_value == 0:
        return cost_of_equity # Default to cost of equity if no value/debt context
    
    weight_equity = market_cap / total_value
    weight_debt = total_debt / total_value
    
    wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
    return wacc

def calculate_dcf(
    fcf_base: float,
    growth_rates: Union[float, List[float]],
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float,
    cash: float = 0,
    debt: float = 0
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Performs a 5-year Discounted Cash Flow (DCF) analysis.
    
    Args:
        fcf_base: Starting Free Cash Flow (Year 0).
        growth_rates: List of 5 growth rates or a single rate for all years.
        wacc: Discount rate.
        terminal_growth: Rate for the Gordon Growth Model (Perpetuity).
        shares_outstanding: Total number of shares.
        cash: Cash and cash equivalents to add if calculating share price from Enterprise Value.
        debt: Total debt to subtract if calculating share price from Enterprise Value.
        
    Returns:
        Dictionary with valuation result and sensitivity analysis.
    """
    if isinstance(growth_rates, (int, float)):
        growth_rates = [growth_rates] * 5
    
    def run_projection(g_rates: List[float], discount_rate: float, t_growth: float) -> float:
        # Prevent division by zero or negative denominator in Gordon Growth
        safe_discount = max(discount_rate, t_growth + 0.001)
        
        projected_fcf = []
        current_fcf = fcf_base
        for g in g_rates:
            current_fcf *= (1 + g)
            projected_fcf.append(current_fcf)
            
        # Terminal Value
        terminal_value = (projected_fcf[-1] * (1 + t_growth)) / (safe_discount - t_growth)
        
        # Present Value
        pv_fcf = sum([fcf / (1 + safe_discount)**(i + 1) for i, fcf in enumerate(projected_fcf)])
        pv_terminal = terminal_value / (1 + safe_discount)**len(projected_fcf)
        
        return pv_fcf + pv_terminal

    enterprise_value = run_projection(growth_rates, wacc, terminal_growth)
    equity_value = enterprise_value + cash - debt
    implied_price = equity_value / shares_outstanding if shares_outstanding > 0 else 0
    
    # Sensitivity Analysis (+/- 1% WACC and Terminal Growth)
    sensitivity = {
        "wacc_plus_1pct": (run_projection(growth_rates, wacc + 0.01, terminal_growth) + cash - debt) / shares_outstanding if shares_outstanding > 0 else 0,
        "wacc_minus_1pct": (run_projection(growth_rates, wacc - 0.01, terminal_growth) + cash - debt) / shares_outstanding if shares_outstanding > 0 else 0,
        "growth_plus_1pct": (run_projection(growth_rates, wacc, terminal_growth + 0.01) + cash - debt) / shares_outstanding if shares_outstanding > 0 else 0,
        "growth_minus_1pct": (run_projection(growth_rates, wacc, terminal_growth - 0.01) + cash - debt) / shares_outstanding if shares_outstanding > 0 else 0,
    }
    
    return {
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "implied_share_price": implied_price,
        "sensitivity": sensitivity
    }

def calculate_multiples_valuation(
    target_metrics: Dict[str, float],
    peer_multiples: Dict[str, List[float]],
    shares_outstanding: float
) -> Dict[str, float]:
    """
    Calculates valuation based on peer benchmarks.
    
    Args:
        target_metrics: Dict with keys 'earnings', 'sales', 'ebitda', 'net_income_growth'.
        peer_multiples: Dict with keys 'pe', 'ps', 'ev_ebitda', 'peg'.
        shares_outstanding: Total shares for per-share calculations.
    """
    results = {}
    
    # PE based valuation
    if "pe" in peer_multiples and "earnings" in target_metrics:
        median_pe = np.median(peer_multiples["pe"]) if peer_multiples["pe"] else 0
        implied_equity_val = median_pe * target_metrics["earnings"]
        results["pe_implied_price"] = implied_equity_val / shares_outstanding if shares_outstanding > 0 else 0
        
    # PS based valuation
    if "ps" in peer_multiples and "sales" in target_metrics:
        median_ps = np.median(peer_multiples["ps"]) if peer_multiples["ps"] else 0
        implied_equity_val = median_ps * target_metrics["sales"]
        results["ps_implied_price"] = implied_equity_val / shares_outstanding if shares_outstanding > 0 else 0
        
    # EV/EBITDA based valuation
    if "ev_ebitda" in peer_multiples and "ebitda" in target_metrics:
        median_ev_ebitda = np.median(peer_multiples["ev_ebitda"]) if peer_multiples["ev_ebitda"] else 0
        # This usually yields Enterprise Value
        implied_ev = median_ev_ebitda * target_metrics["ebitda"]
        # Note: In a real app we'd need net debt to get equity value
        results["ev_ebitda_implied_ev"] = implied_ev
        results["ev_ebitda_implied_price"] = implied_ev / shares_outstanding if shares_outstanding > 0 else 0

    # PEG based valuation
    if "pe" in peer_multiples and "net_income_growth" in target_metrics:
        # PEG = PE / Growth. So Implied PE = PEG * Growth
        if "peg" in peer_multiples:
            median_peg = np.median(peer_multiples["peg"]) if peer_multiples["peg"] else 0
            target_growth = target_metrics["net_income_growth"] * 100 # usually growth is expressed as integer in PEG (e.g. 15 for 15%)
            implied_pe = median_peg * target_growth
            implied_equity_val = implied_pe * target_metrics["earnings"]
            results["peg_implied_price"] = implied_equity_val / shares_outstanding if shares_outstanding > 0 else 0

    return results

def calculate_nav(
    total_assets: float,
    total_liabilities: float,
    intangible_assets: float,
    shares_outstanding: float
) -> Dict[str, float]:
    """Calculates Net Asset Value."""
    nav = (total_assets - intangible_assets) - total_liabilities
    return {
        "nav": nav,
        "nav_per_share": nav / shares_outstanding if shares_outstanding > 0 else 0
    }

def calculate_liquidation_value(
    assets_dict: Dict[str, float],
    total_liabilities: float,
    haircuts: Dict[str, float],
    shares_outstanding: float
) -> Dict[str, float]:
    """
    Calculates Liquidation Value by applying discounts (haircuts) to assets.
    
    Args:
        assets_dict: {'cash': 100, 'receivables': 200, 'inventory': 300, 'fixed_assets': 500}
        total_liabilities: Total liabilities to subtract.
        haircuts: {'cash': 1.0, 'receivables': 0.8, 'inventory': 0.5, 'fixed_assets': 0.3}
        shares_outstanding: Shares for per-share calculation.
    """
    liquidated_assets = 0
    for category, value in assets_dict.items():
        discount = haircuts.get(category, 1.0) # Default to 1.0 (no discount) if not provided
        liquidated_assets += value * discount
        
    liquidation_value = liquidated_assets - total_liabilities
    
    return {
        "liquidation_value": liquidation_value,
        "liquidation_value_per_share": liquidation_value / shares_outstanding if shares_outstanding > 0 else 0
    }
