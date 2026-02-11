from langgraph.graph import StateGraph, START, END
from agent.state import ValuationState
from agent.nodes import (
    input_guardrail_node,
    data_retrieval_node,
    assumption_recommender_node,
    financial_engine_node,
    analysis_synthesis_node
)

def route_after_guardrail(state: ValuationState):
    """Routes to retrieval if valid, otherwise ends with error."""
    if state.get("errors"):
        return "END"
    return "data_retrieval"

def route_after_retrieval(state: ValuationState):
    """Routes to recommender if data found, otherwise ends."""
    if state.get("errors"):
        return "END"
    return "assumption_recommender"

def create_graph():
    """
    Creates the LangGraph state machine for the Valuation Agent.
    """
    workflow = StateGraph(ValuationState)

    # 1. Define Nodes
    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("data_retrieval", data_retrieval_node)
    workflow.add_node("assumption_recommender", assumption_recommender_node)
    workflow.add_node("financial_engine", financial_engine_node)
    workflow.add_node("analysis_synthesis", analysis_synthesis_node)

    # 2. Define Edges & Workflow
    workflow.add_edge(START, "input_guardrail")
    
    # Conditional routing to handle errors early
    workflow.add_conditional_edges(
        "input_guardrail", 
        route_after_guardrail,
        {
            "data_retrieval": "data_retrieval",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "data_retrieval",
        route_after_retrieval,
        {
            "assumption_recommender": "assumption_recommender",
            "END": END
        }
    )

    # Linear flow for the engine and synthesis
    workflow.add_edge("assumption_recommender", "financial_engine")
    workflow.add_edge("financial_engine", "analysis_synthesis")
    workflow.add_edge("analysis_synthesis", END)

    # Compile the graph
    app = workflow.compile()
    return app

# Initialize the agent
valuation_agent = create_graph()
