import os
import streamlit as st
from rag_pipeline import initialize_rag_pipeline

# Configure the Streamlit page layout
st.set_page_config(page_title="FinRAG-AI Chatbot", page_icon="📊", layout="wide")

st.title(" FinRAG-AI: Financial Document Chatbot")
st.markdown("### Powered by LangChain, FAISS, and Local Flan-T5")
st.write(
    "Upload a financial PDF report to begin asking contextual questions without data leaving your machine.")

# Sidebar for document uploading and status tracking
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload Financial PDF", type=["pdf"])

    # Track pipeline state using Streamlit session state
    if uploaded_file is not None:
        # Save the uploaded file locally so PyPDFLoader can read it
        temp_pdf_path = os.path.join(".", uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize and build the RAG pipeline if it hasn't been built for this file yet
        if "rag_chain" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
            with st.spinner("Processing PDF, creating chunks, and indexing into FAISS vector store..."):
                try:
                    # Run your backend pipeline
                    st.session_state.rag_chain = initialize_rag_pipeline(temp_pdf_path)
                    st.session_state.current_file = uploaded_file.name
                    st.success(" Document indexed successfully!")
                except Exception as e:
                    st.error(f"Error initializing pipeline: {str(e)}")

# Chat interface logic
if "rag_chain" in st.session_state:
    st.divider()
    st.subheader(f" Chatting with: {st.session_state.current_file}")

    # Initialize message history tracking
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior conversation turns
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Accept new user queries
    if user_query := st.chat_input("Ask a question about financial highlights or risk disclosures..."):
        # Display the human user question
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Generate the bot's response using the local RAG model
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(user_query)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error during execution: {str(e)}")
else:
    st.info("💡 Please upload a financial PDF report in the sidebar to unlock the chatbot interface.")