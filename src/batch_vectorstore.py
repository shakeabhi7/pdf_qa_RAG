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
    Embeds chunks in batches and builds a FAISS vectorstore.
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

        try:
            vectorstore = batch_embedding(vectorstore,batch,embeddings,batch_num)

        except Exception:
            # ALL retries for this batch failed.
            if save and vectorstore is not None:
                vectorstore.save_local(config.FAISS_INDEX_DIR)
                print(f"Progress saved tp '{config.FAISS_INDEX_DIR}'"
                      f"({batch_num - 1}/{total_batches} batches completed before failure.)")
                raise
        # Save progress after every successfull batch
        if save:
            vectorstore.save_local(config.FAISS_INDEX_DIR)

        # Don't sleep after the very last batch
        if batch_num < total_batches:
            print(f" Waiting {DELAY}s before next batch")
            time.sleep(DELAY)
    print("[OK] All batches embedded, vectorstore built")
    if save:
        print(f"Final vectorstore saved to '{config.FAISS_INDEX_DIR}'")
    return vectorstore


def load_vectorstore():
    """
    Loads a previously saved FAISS vectorstore from disk.
    """

    embeddings = get_embeddings()

    if not os.path.exists(config.FAISS_INDEX_DIR):
        raise FileNotFoundError(
            f"No Saved vectorstore found at '{config.FAISS_INDEX_DIR}'. "
            "Run build_vectorstore() first."
        )
    vectorstore = FAISS.load_local(
        config.FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"[OK] Vectorstore loaded from '{config.FAISS_INDEX_DIR}' ")
    return vectorstore


def run_test_query(vectorstore,query):
    """Runs a single retrieval test against an already built vectorstore"""
    retriever = vectorstore.as_retriever(search_kwargs={"k":config.RETRIEVAL_K})
    results = retriever.invoke(query)

    print(f"\n--- Retrieved {len(results)} chunks for query: '{query}' ---\n")
    for i, doc in enumerate(results):
        print(f"Result {i+1} (page {doc.metadata.get('page', '?')}):")
        print(doc.page_content[:400])
        print("-" * 50)



if __name__ == "__main__":
    """
    This Only Builds(and re-embeds) when there's no saved index yet.
    If faiss_index/ already exists, it just loads it.
    """
    if os.path.exists(config.FAISS_INDEX_DIR):
        print("Found existing vectorstore, loading from disk(no re-embedding)...")
        print("If you changed the PDF, delete the faiss_index/ folder first and re-run.")

        vectorstore = load_vectorstore()
    else:
        print("No saved vectorstore found, building from scratch...")
        from src.load_split import load_pdf,split_documents

        pdf_path = os.path.join(config.DATA_DIR,"Data_Science_from_Scratch_1-109.pdf")
        docs = load_pdf(pdf_path)
        chunks = split_documents(docs)
        vectorstore = build_vectorstore(chunks)

    # --- Manual retrieval test ---
    test_query = "Tell me about Statistics."
    run_test_query(vectorstore,test_query)


    