from langchain_google_ai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src import config
from src.vectorstore import load_vectorstore

# The prompt tells the LLM exactly how to behave:
# - Only use the provided context
# - Don't make things up if the answer isn't there

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful study assistant. Answer the question using ONLY the
context below. If the answer is not present in the context, say clearly
that it is not available in the notes — do not make up an answer.
 
Context:
{context}
 
Question: {question}
 
Answer:"""
)

def format_docs(docs):
    """
    Combine the retrieved chunks into single text block, with page numbers
    so we can trace where each piece of context came from
    """

    formatted = []
    for doc in docs:
        page = docs.metadata.get("page","?")
        formatted.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(vectorstore):
    """
    Build the full RAG chain using LCEL:
    Retriever -> format context -> prompt -> LLM -> parse output as plain text
    """

    retriever = vectorstore.as_retriever(search_kwargs={"k":config.RETRIEVAL_K})

    llm = ChatGoogleGenerativeAI()
