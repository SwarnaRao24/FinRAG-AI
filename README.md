# FinRAG-AI: Advanced Financial Document Chatbot

FinRAG-AI is a powerful, production-ready Retrieval-Augmented Generation (RAG) pipeline optimized for analyzing complex financial documents, such as 10-K filings, annual reports, and quarterly statements. 

By combining a high-performance local vector database (**FAISS**) with cloud-based synthesis via **Google AI Studio (Gemini 2.5 Flash)**, the application delivers incredibly fast, contextual financial insights through an intuitive web interface.

##  Key Features
* **Financial Document Processing:** Seamlessly handles massive, multi-page financial text layouts.
* **Hybrid Architecture:** Local chunking and embedding storage ensure zero data indexing overhead, paired with Google's high-speed cloud generation.
* **Advanced Financial Synthesis:** Leverages Gemini 2.5 Flash to extract intricate numbers, provisions, risks, and trends without text length bottlenecks.
* **Interactive Chat Dashboard:** A clean user interface built with Streamlit featuring file uploads and sequential chat histories.

---

## ️ Technical Architecture Blueprint

The system coordinates data ingestion, indexing, and inference through a series of logical modular blocks:

1. **Document Ingestion:** Raw PDF data is parsed into memory using `PyPDFLoader`.
2. **Context Chunking:** Text is broken down into structured $1000$-word segments with a $100$-word overlap using `RecursiveCharacterTextSplitter` to protect cross-sentence continuity.
3. **Vector Embeddings:** Tokens are translated into high-dimensional vectors via Hugging Face's open-source `all-MiniLM-L6-v2` transformer model.
4. **Local Indexing:** Embeddings are written to a lightning-fast local `FAISS` vector store database.
5. **Retrieval Chain:** When a query is asked, FAISS runs a similarity search to isolate the top $k=5$ mathematically relevant context pieces.
6. **Inference Pipeline:** Context and prompts are executed sequentially using LangChain Expression Language (LCEL) and generated via the Google Gemini API.

---

##  Tech Stack & Dependencies
* **Framework Orchestration:** LangChain, LangChain Community, LangChain Google GenAI
* **Vector Store Indexing:** FAISS (CPU variant)
* **LLM Foundation:** Google AI Studio API (`gemini-2.5-flash`)
* **Embedding Model:** Hugging Face Transformers (`all-MiniLM-L6-v2`)
* **Framework Infrastructure:** PyTorch & TorchVision
* **User Interface:** Streamlit

---

## ️ Setup & Installation

Follow these steps to configure the application environment on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/SwarnaRao24/FinRAG-AI.git](https://github.com/SwarnaRao24/FinRAG-AI.git)
cd FinRAG-A
```
---
**Developer:** Swarna Rao  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/swarnamukhi-chintalapudi)

**Focus:** Advanced GenAI | AI Full-stack | NLP | Fintech
