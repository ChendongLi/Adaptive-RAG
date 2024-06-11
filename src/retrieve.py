from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from utils.gcp import embed_text
from langsmith import traceable

# TODO config model name


class Retriever:
    """
    Build vectordb in Chroma 
    Embedding in VertexAI
    Retrieve similar documents
    """

    # Document URLs
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    ]

    def __init__(self):
        self.k = 3
        self.emb_model_name = "text-embedding-004"
        self.collection_name = "adp-rag-chroma"
        self.persist_directory = "data/db/chroma_db"
        self.embd = VertexAIEmbeddings(model_name=self.emb_model_name)

    def process_docs(self) -> list:
        """
        Load documents from the URLs
        Split the documents into chunks
        Return: document chunks
        """
        docs = [WebBaseLoader(url).load() for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        return doc_splits

    def create_db(self) -> Chroma:
        """
        Create a new vectordb in Chroma
        Return: vectorstore
        """
        doc_splits = self.process_docs()
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=self.collection_name,
            embedding=self.embd,
            persist_directory=self.persist_directory
        )

        return vectorstore

    @traceable
    def retrieve(self) -> Chroma:
        """
        Load Chroma DB from Disk and Return Retriever
        Return: Retriever
        """
        vectorstore = Chroma(persist_directory=self.persist_directory,
                             collection_name=self.collection_name, embedding_function=self.embd)

        if len(vectorstore) == 0:
            print('No documents in the vectorstore, Create a new one')
            vectorstore = self.create_db()

        return vectorstore.as_retriever()
