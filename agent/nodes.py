from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import AgentState
from config import MODEL_ID, API_KEY, tavily_client
from utils import sanitize_content
from rag.vector_store import hybrid_search

def get_llm():
    return ChatGoogleGenerativeAI(model=MODEL_ID, google_api_key=API_KEY)

def route_initial_query(state: AgentState):
    print("--- NODE: INITIAL ROUTER ---")
    llm = get_llm()
    query = state["query"]
    
    prompt = f"""You are an intelligent routing agent for an academic research assistant. 
    Analyze the user's input: '{query}'
    
    Does this query require searching an external database of academic research papers to answer accurately?
    - If it is asking about specific concepts, technical details, authors, or theories, respond with 'search'.
    - If it is a greeting, general chat, asking about your capabilities, or a request to summarize the conversation history, respond with 'chat'.
    
    Respond strictly with a single word: 'search' or 'chat'."""
    
    try:
        raw_content = llm.invoke(prompt).content
        decision = sanitize_content(raw_content).strip().lower()
    except Exception as e:
        print(f"Routing error: {e}")
        decision = "search" 

    if "search" in decision:
        print("> Decision: Route to Vector Retrieval")
        return "retrieve"
    else:
        print("> Decision: Route to General Chat")
        return "generate"

def retrieve_node(state: AgentState):
    print("--- NODE: RETRIEVE FROM VECTOR DB ---")
    query_str = state['query']
    llm = get_llm()
    
    try:
        expansion_prompt = f"Rewrite this query to include technical keywords for vector search. Return ONLY the query.\nQuery: {query_str}"
        raw_content = llm.invoke(expansion_prompt).content
        expanded_query = sanitize_content(raw_content).strip()
    except Exception as e:
        print(f"Expansion error: {e}")
        expanded_query = query_str

    # Fetch using the modularized function
    final_docs = hybrid_search(expanded_query, top_k=5)
    
    return {"documents": final_docs, "loop_count": state.get("loop_count", 0) + 1}

def grade_documents_node(state: AgentState):
    print("--- NODE: GRADE DOCUMENT RELEVANCE ---")
    llm = get_llm()
    docs = state["documents"]
    
    if not docs:
        return {"web_fallback": True}
        
    prompt = f"Does ANY of the following documents contain keywords or semantic meaning related to the question: '{state['query']}'?\nDocuments: {[d['content'] for d in docs]}\nRespond ONLY with 'yes' or 'no'."
    
    raw_content = llm.invoke(prompt).content
    response = sanitize_content(raw_content).strip().lower()
    
    return {"web_fallback": False} if "yes" in response else {"web_fallback": True}

def rewrite_query_node(state: AgentState):
    print("--- NODE: REWRITE QUERY ---")
    llm = get_llm()
    prompt = f"The original query failed to retrieve relevant documents. Rewrite the query to be more effective for a vector database: {state['query']}\nReturn ONLY the rewritten query."
    
    raw_content = llm.invoke(prompt).content
    new_query = sanitize_content(raw_content).strip()
    return {"query": new_query}

def web_search_node(state: AgentState):
    print("--- NODE: WEB SEARCH FALLBACK ---")
    try:
        tavily_results = tavily_client.search(f"arXiv academic paper {state['query']}")
        docs = [{"content": r["content"], "meta": {"source": r["url"], "page": "Web"}} for r in tavily_results.get("results", [])]
        return {"documents": docs}
    except Exception as e:
        return {"documents": []}

def generate_node(state: AgentState):
    print("--- NODE: GENERATE ANSWER ---")
    llm = get_llm()
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in state.get('chat_history', [])[-4:]]) if state.get('chat_history') else "No previous history."
    context_str = "\n\n".join([f"Source: {d['meta'].get('source', 'Unknown')}\n{d['content']}" for d in state.get("documents", [])]) if state.get("documents") else "No relevant information found."
    
    prompt = f"""You are a helpful, expert academic research assistant. 
    
    Conversation History:
    {history_str}
    
    Retrieved Context (if any):
    {context_str}
    
    Latest User Query: {state['query']}
    
    Instructions:
    1. If the user is just saying hello or making general conversation, respond naturally and offer your help with research.
    2. If the user asks a research question, base your answer primarily on the Retrieved Context.
    """
    try:
        raw_content = llm.invoke(prompt).content
        response = sanitize_content(raw_content)
        citations = set([f"Source: {d['meta'].get('source', 'Unknown')}, Page: {d['meta'].get('page', 'N/A')}" for d in state.get("documents", [])])
        citation_str = "\n\n**Sources:**\n" + "\n".join([f"* {c}" for c in citations]) if citations else "\n\n*No internal sources used.*"
        return {"answer": response + citation_str}
    except Exception as e:
        return {"answer": f"Error generating response: {e}"}

def decide_to_generate(state: AgentState):
    if state.get("web_fallback"):
        if state.get("loop_count", 0) > 1: return "web_search"
        return "rewrite_query"
    return "generate"