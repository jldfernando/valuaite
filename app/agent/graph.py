from langgraph.graph import StateGraph, START, END
from agent.state import ValuationState
from agent.nodes import (
    ticker_extractor_node,
    data_retrieval_node,
    analyst_planner_node,
    financial_engine_node,
    analysis_synthesis_node
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

def create_graph():
    """
    Creates the LangGraph state machine for the Valuation Agent.
    """
    workflow = StateGraph(ValuationState)

    # 1. Define Nodes
    workflow.add_node("ticker_extractor", ticker_extractor_node)
    workflow.add_node("data_retrieval", data_retrieval_node)
    workflow.add_node("analyst_planner", analyst_planner_node)
    workflow.add_node("financial_engine", financial_engine_node)
    workflow.add_node("analysis_synthesis", analysis_synthesis_node)

    # 2. Define Edges & Workflow
    workflow.add_edge(START, "ticker_extractor")
    
    # Conditional routing to handle errors early
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

    # Linear flow for the brain and math
    workflow.add_edge("analyst_planner", "financial_engine")
    workflow.add_edge("financial_engine", "analysis_synthesis")
    workflow.add_edge("analysis_synthesis", END)

    # Compile the graph
    app = workflow.compile()
    return app

# Initialize the agent
valuation_agent = create_graph()
