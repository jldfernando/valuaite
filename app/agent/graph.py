from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import ValuationState
from agent.nodes import (
    ticker_extractor_node,
    data_retrieval_node,
    analyst_planner_node,
    financial_engine_node,
    analysis_synthesis_node,
    scenario_analysis_node
)

def route_after_extraction(state: ValuationState):
    """Routes to retrieval if a ticker was found, otherwise ends."""
    if state.get("errors") or state.get("ticker") == "UNKNOWN":
        return "END"
    return "data_retrieval"

def route_after_retrieval(state: ValuationState):
    """Routes to planner if data found, otherwise ends."""
    if state.get("errors"):
        return "END"
    return "analyst_planner"

def route_after_planner(state: ValuationState):
    """
    Routes to engine for full valuation, or directly to synthesis for quick inquiries.
    Added keyword safety check and 'REVISE' support for HITL negotiation.
    """
    if state.get("errors"):
        return "END"
    
    # 0. Check for Rewind/Negotiation signal (set by UI)
    if state.get("current_step") == "analyst_planner":
        print(">>> Negotiation detected. Returning to analyst_planner.")
        return "analyst_planner"

    # 1. Check Intent from the planner
    intent = state.get("assumptions", {}).get("intent", "FULL_VALUATION")
    
    # 2. Safety Check: If user asks for specific math but LLM said QUICK_INQUIRY
    user_query = ""
    if state.get("messages"):
        last_msg = state["messages"][-1]
        user_query = (last_msg["content"] if isinstance(last_msg, dict) else last_msg.content).lower()

    valuation_keywords = ["valuate", "value", "worth", "dcf", "nav", "pe", "ratio", "multiple", "multiplier", "intrinsic", "fair price", "liquidation"]
    if intent == "QUICK_INQUIRY" and any(k in user_query for k in valuation_keywords):
        print(">>> Safety Check: Valuation keywords detected despite QUICK_INQUIRY intent. Overriding to FULL_VALUATION.")
        intent = "FULL_VALUATION"

    if intent == "QUICK_INQUIRY":
        print(">>> Quick Inquiry confirmed. Skipping math engine.")
        return "analysis_synthesis"
    
    return "financial_engine"

def route_after_synthesis(state: ValuationState):
    """Allows jumping to scenario analysis from the final state."""
    if state.get("current_step") == "scenario_analysis":
        return "scenario_analysis"
    return "END"

def create_graph():
    """
    Creates the LangGraph state machine with branching and memory.
    """
    workflow = StateGraph(ValuationState)

    # 1. Define Nodes
    workflow.add_node("ticker_extractor", ticker_extractor_node)
    workflow.add_node("data_retrieval", data_retrieval_node)
    workflow.add_node("analyst_planner", analyst_planner_node)
    workflow.add_node("financial_engine", financial_engine_node)
    workflow.add_node("analysis_synthesis", analysis_synthesis_node)
    workflow.add_node("scenario_analysis", scenario_analysis_node)

    # 2. Define Edges & Workflow
    workflow.add_edge(START, "ticker_extractor")
    
    workflow.add_conditional_edges(
        "ticker_extractor", 
        route_after_extraction,
        {
            "data_retrieval": "data_retrieval",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "data_retrieval",
        route_after_retrieval,
        {
            "analyst_planner": "analyst_planner",
            "END": END
        }
    )

    # Branching logic after the brain
    workflow.add_conditional_edges(
        "analyst_planner",
        route_after_planner,
        {
            "financial_engine": "financial_engine",
            "analysis_synthesis": "analysis_synthesis",
            "analyst_planner": "analyst_planner",
            "END": END
        }
    )
    
    # Transitions
    workflow.add_edge("financial_engine", "analysis_synthesis")
    
    workflow.add_conditional_edges(
        "analysis_synthesis",
        route_after_synthesis,
        {
            "scenario_analysis": "scenario_analysis",
            "END": END
        }
    )
    
    workflow.add_edge("scenario_analysis", END)

    # 3. Add Persistence (Memory)
    memory = MemorySaver()
    
    # Compile the graph with an interrupt before math begins
    # This is the "Blueprint Approval" stop
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["financial_engine"]
    )
    return app

# Initialize the agent
valuation_agent = create_graph()
