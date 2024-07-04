import yaml
import os
from pprint import pprint
from src.graph import build_adaptive_graph, build_self_rag_graph

with open("secrets/secrets.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

os.environ["LANGCHAIN_TRACING_V2"] = config['langsmith']['cisco']['langchain_tracing_v2']
os.environ["LANGCHAIN_ENDPOINT"] = config['langsmith']['cisco']['langchain_endpoint']
os.environ["LANGCHAIN_PROJECT"] = config['langsmith']['cisco']['langchain_project']
os.environ["LANGCHAIN_API_KEY"] = config['langsmith']['cisco']['langchain_api_key']


# retrive
# retriever = Retriever().retrieve()
# print(retriever.invoke(
#     input={"question": "What player at the Bears expected to draft first in the 2024 NFL draft"}))

# ------------------------------------------------------

def agent_graph_finite_loop(question: str, adaptive_graph: bool = False):
    if adaptive_graph:
        graph = build_adaptive_graph()
    else:
        graph = build_self_rag_graph()

    inputs = {"question": question}
    thread = {"configurable": {"thread_id": "1"}}

    for output in graph.stream(inputs, thread):
        for key, value in output.items():
            pprint(f"[Node]: {key}")

    if len(graph.get_state(thread).next) > 0 and graph.get_state(thread).next[0] == 'retrieve':
        pprint("---FIRST INTERRUPT---")
        pprint(
            f"question: {graph.get_state(thread).values['question']}\nnext action: {graph.get_state(thread).next}")

        for output in graph.stream(None, thread):
            for key, value in output.items():
                pprint(f"[Node]: {key}")

    if len(graph.get_state(thread).next) > 0 and graph.get_state(thread).next[0] == 'retrieve':
        pprint("---SECOND INTERRUPT BEFORE UPDATE STATE---")
        pprint(
            f"question: {graph.get_state(thread).values['question']}\nnext action: {graph.get_state(thread).next}")

        current_state = graph.get_state(thread)
        current_state.values['force_generate'] = True
        graph.update_state(thread, current_state.values)

        pprint("---SECOND INTERRUPT AFTER UPDATE STATE---")
        pprint(
            f"question: {graph.get_state(thread).values['question']}\nnext action: {graph.get_state(thread).next}\nforece_generate: {graph.get_state(thread).values['force_generate']}")

        for output in graph.stream(None, thread):
            for key, value in output.items():
                pprint(f"[Node]: {key}")

        pprint(
            f"documents: {value['question']}\nquestion: {value['question']}\ngeneration: {value['generation']}")

    else:
        pprint(
            f"documents: {value['question']}\nquestion: {value['question']}\ngeneration: {value['generation']}")


if __name__ == "__main__":

    # agent_graph_finite_loop(
    #     "What are the types of agent memory?", adaptive_graph=True)

    agent_graph_finite_loop(
        "What player at the Bears expected to draft first in the 2024 NFL draft?", adaptive_graph=True)
