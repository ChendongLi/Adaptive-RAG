import yaml
import os
from pprint import pprint
from utils.langchain_llm import LangchainCiscoGPT4, LangchainGermini
from src.retrieve import Retriever
from src.router import question_router, route_question
from src.grader import grade_documents
from src.generate import generate_answer
from src.rewriter import transform_query
from src.search import web_search
from src.graph import build_graph

from utils.gcp import embed_text

with open("secrets/secrets.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

os.environ["LANGCHAIN_TRACING_V2"] = config['langsmith']['cisco']['langchain_tracing_v2']
os.environ["LANGCHAIN_ENDPOINT"] = config['langsmith']['cisco']['langchain_endpoint']
os.environ["LANGCHAIN_PROJECT"] = config['langsmith']['cisco']['langchain_project']
os.environ["LANGCHAIN_API_KEY"] = config['langsmith']['cisco']['langchain_api_key']


# retrive
retriever = Retriever().retrieve()
# print(retriever.invoke(input="how to use agent to build agentic rag"))


# queston router
# print(
#     route_question(
#         {"question": "Who will the Bears draft first in the NFL draft?"}
#     )
# )
# print(route_question(
#     {"question": "What are the types of agent memory?"}))

# grade documents
# question = "agent memory"
# docs = retriever.get_relevant_documents(question)
# print(grade_documents({"question": question, "documents": docs}))


# generate answer
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# docs = format_docs(docs)
# print(generate_answer({"question": question, "documents": docs}))


# question rewriter
# print(transform_query({"question": question, "documents": docs}))

# web search
# print(web_search({"question": question}))


# build graph

graph = build_graph()

# Run
inputs = {
    "question": "What player at the Bears expected to draft first in the 2024 NFL draft?"
}
for output in graph.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])
