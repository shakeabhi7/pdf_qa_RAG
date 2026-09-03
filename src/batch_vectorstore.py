import os
import time
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src import config

BATCH_SIZE = 80
DELAY= 60
MAX_RETRIES = 3
RETRY_DELAY = 30

def get_embeddings():
    """
    Returns a Gemini embeddings object.
    This converts text into numeric vectors that cpature semantic meaning.
    """

    return GoogleGenerativeAIEmbeddings(
        model = config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY
    )

def batch_embedding(vectorstore,batch,embeddings,batch_num):
    """
    Tries to embed a batch of chunks. Retries with delay.
    Returns the updated vectorstore.
    """
    for attempt in range(1,MAX_RETRIES+1):
        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch,embeddings)
            else:
                vectorstore.add_documents(batch)
            print(f"[OK] Batch {batch_num} embedded ({len(batch)} chunks)")
            return vectorstore
        
        except Exception as e:
            print(f"Batch {batch_num} failed on attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                print(f"Waiting {RETRY_DELAY}s before retrying...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Batch {batch_num} failed after {MAX_RETRIES} attemps. skipping.")
                raise

    return vectorstore


def build_vectorstore(chunks,save=True):
    """
    Embeds chunks in batches and buils a FAISS vectorstore.
    Saves the index to disk when done
    """

    embeddings = get_embeddings()
    vectorstore = None

    total_batches = (len(chunks) + BATCH_SIZE - 1)// BATCH_SIZE

    print(f" Embedding {len(chunks)} chunks in {total_batches} batches of {BATCH_SIZE}")

    if save:
        os.makedirs(config.FAISS_INDEX_DIR,exist_ok=True)
    for i in range(0,len(chunks),BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = chunks[i:i+BATCH_SIZE]

def load_vectorstore():
    """
    Loads a previously saved FAISS vectorstore from disk.
    """

    embeddings = get_embeddings()

    if not os.path.exists(config.FAISS_INDEX_DIR):
        raise FileNotFoundError(
            f"No Saved vectorstore found at '{config.FAISS_INDEX_DIR}'"
            "Run build_vectorstore() first."
        )
    vectorstore = FAISS.load_local(
        config.FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"[OK] Vectorstore loaded from '{config.FAISS_INDEX_DIR}' ")
    return vectorstore

if __name__ == "__main__":
    from src.load_split import load_pdf,split_documents

    pdf_path = os.path.join(config.DATA_DIR,"Data_Science_from_Scratch_1-109.pdf")

    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)

    # -- Manual test
    retriever = vectorstore.as_retriever(search_kawrgs={"k":config.RETRIEVAL_K})

    test_query = "Tell mein about the Modelling and correctness from Machine Learning."

    results = retriever.invoke(test_query)

    print(f"\n--- Retrieved {len(results)} chunks for query: '{test_query}' ---\n")
    for i, doc in enumerate(results):
        print(f"Result {i+1} (page {doc.metadata.get('page', '?')}):")
        print(doc.page_content[:400])
        print("-" * 50)