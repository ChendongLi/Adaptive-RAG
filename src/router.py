# Router
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from utils.langchain_llm import LangchainCiscoGPT4
from src.chain import chain_creator

# Data model


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


def question_router():
    """
    Route a user question to select a vectorstore or web search.
    """
    llm = LangchainCiscoGPT4().cisco_gpt4()

    # Prompt
    system = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on the topics about agents, prompt engineering, and adversarial attacks. Otherwise, use web-search."""

    human = "{question}"

    return chain_creator(llm, system, human, RouteQuery)


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router().invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
