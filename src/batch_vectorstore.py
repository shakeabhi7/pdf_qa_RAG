import os
import time
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src import config

BATCH_SIZE = 80
DELAY_= 60
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
    