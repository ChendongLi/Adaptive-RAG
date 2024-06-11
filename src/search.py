import os
import yaml
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

with open("secrets/secrets.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

os.environ["TAVILY_API_KEY"] = config['tavily']['api_key']


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    # Web search
    web_search_tool = TavilySearchResults(k=1)

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}
