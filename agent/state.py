from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    query: str
    chat_history: List[dict]
    documents: List[Dict[str, Any]]
    answer: str
    web_fallback: bool
    loop_count: int