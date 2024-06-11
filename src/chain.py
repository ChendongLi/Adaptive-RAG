from langchain_core.prompts import ChatPromptTemplate


def chain_creator(llm: object, system: str, human: str, output_parse: object) -> object:
    """
    Create a chain of LLMs to process a user query
    Args:
        llm: LLM model
        system: system message
        human: user message
        output_parse: Data Model Class for structured output
    Returns:
        Langchain LCEL object
    """
    structured_llm_output = llm.with_structured_output(output_parse)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ])

    return prompt | structured_llm_output
