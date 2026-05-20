import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Import the Google AI Studio LLM wrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def initialize_rag_pipeline(pdf_path):
    print(" Step 1 & 2: Loading PDF and chunking text...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # We can use large, comprehensive chunks because Gemini handles context easily
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(docs)
    print(f" Created {len(final_documents)} text chunks.")

    print(" Step 3 & 4: Generating embeddings and creating FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(final_documents, embeddings)

    # Retrieve top 5 highly relevant text chunks or can be adjusted
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print(" FAISS index created successfully.")

    print(" Step 5: Connecting to Google AI Studio (Gemini 1.5 Flash)...")

    #Replace your API key
    api_key = os.getenv("GOOGLE_API_KEY", "GOOGLE_API_KEY")

    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )

    system_prompt = (
        "You are an expert financial analyst assistant. Use the following retrieved document context "
        "to answer the question thoroughly. If the context does not explicitly contain the answer, "
        "use your analytical capacity to infer it based on what is available, or guide the user on what "
        "is missing. Keep answers professional, crisp, and contextual.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n\n"
        "Answer:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(system_prompt)
            | llm
            | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    sample_pdf = "sample_financial_report.pdf"
    if os.path.exists(sample_pdf):
        chain = initialize_rag_pipeline(sample_pdf)
        query = "Can you give me a quick summary of the report under 50 words?"
        print(f"\n User asks: {query}")
        response = chain.invoke(query)
        print(f"\n Bot Answer:\n{response}")
    else:
        print(f"\n Please place a sample PDF named '{sample_pdf}' in this directory to run a test.")