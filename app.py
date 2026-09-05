import streamlit as st

from src.vectorstore import load_vectorstore
from src.chain import build_rag_chain, ask

st.set_page_config(page_title="PDF Notes Q&A Bot",page_icon="📚")

st.title("📚 PDF Notes Q&A Bot")
st.caption("Ask questions from the PDF content.")


@st.cache_resource
def get_rag_chain():
    """
    Loads the vectorstore and builds the RAG Chain ONCE, then caches it.
    Without this, Streamlit would reload everything on every single interaction. 
    """
    vectorstore = load_vectorstore()
    return build_rag_chain(vectorstore)


# Build (or fetch cached) chain
try:
    rag_chain = get_rag_chain()
except FileNotFoundError:
    st.error("No Saved vectorstore found. Run 'python -m src.batch_vectorstore' first "
             "to build it from your PDF. ")
    st.stop()


# keep chat history across reruns using Streamlit's session state

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# chat input box (appears pinned at bottom)
question = st.chat_input("Ask a Questions from the PDF..")

if question:
    # Show the user's question immediately
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get and show the answer

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(rag_chain,question)
        st.markdown(answer)

    st.session_state.messages.append({"role":"assistant","content":answer})
    