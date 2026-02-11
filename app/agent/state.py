from typing import Dict, List, Any, Optional, Annotated, TypedDict
import operator

class ValuationState(TypedDict):
    """
    State schema for the Business Valuation AI Agent.
    """
    # 1. Identification
    ticker: str
    
    # 2. Data Storage (populated by Data Retrieval Node)
    company_data: Dict[str, Any]
    peer_data: Dict[str, Any] # Multiples of peers
    
    # 3. Human-in-the-Loop & Assumptions (populated by Assumption Node)
    assumptions: Dict[str, Any]  # WACC, Terminal Growth, Asset Haircuts
    user_adjustments: Dict[str, Any] # Manual overrides from HITL interrupts
    
    # 4. Calculation Outputs (populated by Financial Engine Node)
    valuation_results: Dict[str, Any] # DCF, Multiples, NAV, Liquidation outputs
    
    # 5. Final Report & Feedback
    analysis_report: str
    
    # 6. Graph Flow & Chat History
    # Annotated with operator.add so messages accumulate instead of overwriting
    messages: Annotated[List[Dict[str, str]], operator.add]
    
    # 7. Metadata
    current_step: str
    errors: List[str]
