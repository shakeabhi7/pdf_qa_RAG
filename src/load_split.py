import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src import config

def load_pdf(pdf_path):
    """
    Loads a single PDF file and return a list of langchain Document objects.
    Each page becomes one Document 
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f" Loaded {len(documents)} pages from '{pdf_path}")
    return documents

def split_documents(documents):
    """
    Splits documents into smaller chunks:
    Embeddings and LLM gets only context
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap =config.CHUNK_OVERLAP,
        separators=["\n\n","\n",". ", " ",""], # paragraph first, then lines, sentence
    )

    chunks = splitter.split_documents(documents)
    print(f" Split {len(documents)} pages into {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    # this block only run when this files is executed directly
    pdf_path = os.path.join(config.DATA_DIR,"dl-curriculum.pdf")

    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)

    #print first 2 chunks
    print("\n ---Sample chunks---\n")
    for i, chunk in enumerate(chunks[:2]):
        print(f"Chunk {i+1} (page {chunk.metadata.get('page', '?')}):")
        print(chunk.page_content[:1000])
        