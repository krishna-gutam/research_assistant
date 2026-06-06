from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    route_initial_query, retrieve_node, grade_documents_node,
    rewrite_query_node, web_search_node, generate_node, decide_to_generate
)

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)

    workflow.set_conditional_entry_point(
        route_initial_query,
        {
            "retrieve": "retrieve",
            "generate": "generate"
        }
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents", 
        decide_to_generate, 
        {
            "rewrite_query": "rewrite_query", 
            "web_search": "web_search", 
            "generate": "generate"
        }
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()