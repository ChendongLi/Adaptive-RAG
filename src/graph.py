from typing_extensions import TypedDict
from typing import List

from langgraph.graph import END, StateGraph
from src.retrieve import Retriever
from src.router import question_router, route_question
from src.grader import grade_documents, grade_generation_v_documents_and_question
from src.generate import generate_answer, decide_to_generate
from src.rewriter import transform_query
from src.search import web_search


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


def save_graph(graph: object, graph_name: str):
    try:
        graph.get_graph().draw_mermaid_png(
            output_file_path=f'data/image/{graph_name}.png')
    except Exception as e:
        print(f"Error generate lang graph png {e}")


def build_graph():
    workflow = StateGraph(GraphState)
    retrieve = Retriever().retrieve()

    # Define the nodes
    workflow.add_node("web_search", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate_answer)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query

    # Build graph
    workflow.set_conditional_entry_point(
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile
    graph = workflow.compile()

    save_graph(graph, "adaptive_rag_graph")

    return graph
