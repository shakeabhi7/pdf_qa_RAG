from langchain_google_genai import ChatGoogleGenerativeAI
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
 
Respond in the same language style as the question (if the question is in
Hinglish, answer in Hinglish; if in English, answer in English).
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
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(vectorstore):
    """
    Build the full RAG chain using LCEL:
    Retriever -> format context -> prompt -> LLM -> parse output as plain text
    """

    retriever = vectorstore.as_retriever(search_kwargs={"k":config.RETRIEVAL_K})

    llm = ChatGoogleGenerativeAI(
        model = config.LLM_MODEL,
        google_api_key = config.GOOGLE_API_KEY,
        temperature = 0
    )

    rag_chain = (
        {"context":retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain

def ask(rag_chain,question):
    """Convenince wrapper to run a single question through the chain"""
    return rag_chain.invoke(question)

if __name__ == "__main__":
    # Load the already-sorted vecctorstore

    vectorstore = load_vectorstore()
    rag_chain = build_rag_chain(vectorstore)

    print("\n --- RAG Chatbot (type 'exit to quit) --- \n")

    while True:
        question = input("Your question : ").strip()
        if question.lower() in ("exit","quit"):
            print("Bye!")
            break
        if not question:
            continue

        answer = ask(rag_chain,question)
        print(f"\nAnswer : {answer}\n")
        print("-"*60)

