import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src import config

def get_embeddings():
    """
    Returns a Gemini embeddings object.
    This converts text into numeric vectors that cpature semantic meaning.
    """

    return GoogleGenerativeAIEmbeddings(
        model = config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY
    )

def build_vectorstore(chunks,save=True):
    """
    Takes a list of chunks, embeds them, and builds aFAISS vectorstore.
    Optionally saves the index to disk so we don't have to re-embed every time.
    """
    embeddings = get_embeddings()

    print(f"Embedding {len(chunks)} chunks")
    vectorstore = FAISS.from_documents(chunks,embeddings)
    print("[OK] Vectorstore built")

    if save:
        os.makedirs(config.FAISS_INDEX_DIR,exist_ok=True)
        vectorstore.save_local(config.FAISS_INDEX_DIR)
        print("Vectorstore saved to {config.FAISS_INDEX_DIR}")

    return vectorstore

def load_vectorstore():
    """
    Loads a previously saved FAISS vectorstore from disk.
    Use this instead of rbuilding every time you run the app.
    """
    embeddings = get_embeddings()

    if not os.path.exists(config.FAISS_INDEX_DIR):
        raise FileNotFoundError(
            f"No Saved vectorstore found at '{config.FAISS_INDEX_DIR}'."
            "Run build_vectorstore() first"
        )

    vectorstore = FAISS.load_local(
        config.FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print(f"Vector loaded from '{config.FAISS_INDEX_DIR}' ")
    return vectorstore

if __name__ == "__main__":
    """
    Build the vectorstore
    Test query to check retrieval quality
    """

    from src.load_split import load_pdf, split_documents

    pdf_path = os.path.join(config.DATA_DIR,"dl-curriculum.pdf")

    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)

    # -- Manual retrieval test
    retriever = vectorstore.as_retriever(search_kwargs={"k":config.RETRIEVAL_K})

    test_query = "Tell me about CNN Architectures and Innovations."
    
    results_with_scores = vectorstore.similarity_search_with_score(test_query, k=8)

    print(f"\n--- Top 8 results with similarity scores ---\n")
    for i, (doc, score) in enumerate(results_with_scores):
        print(f"Result {i+1} | Score: {score:.4f} | Page: {doc.metadata.get('page', '?')}")
        print(doc.page_content[:200])
        print("-" * 50)