# ⚖️ Legal Assistant AI

An intelligent, Retrieval-Augmented Generation (RAG) powered legal research assistant. This application allows you to upload legal documents (PDFs) and ask questions in natural language. It provides precise, citation-backed answers by searching through your private document library or the public web.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-%23FF4B4B)](https://streamlit.io/)

## ✨ Features

- **📚 Document Processing**: Upload and process single PDFs or entire folders of legal documents.
- **🔍 Intelligent Search**: Ask questions in natural language and get answers sourced directly from your uploaded case law.
- **🌐 Web Search Fallback**: Toggle web search for questions beyond your uploaded documents.
- **⚡ High Performance**: Powered by Groq's ultra-fast LLama 3.1 API and Pinecone's efficient vector database for quick retrieval.
- **📑 Citation Tracking**: Responses include citations to the original source documents for verification.

## 🚀 Live Demo

A live instance of this application is hosted on Hugging Face Spaces:  
**[https://huggingface.co/spaces/PreetPatelRK30/legal-assistant-ai](https://huggingface.co/spaces/PreetPatelRK30/legal-assistant-ai)**  

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **LLM & Chat**: [LangChain](https://www.langchain.com/) & [Groq](https://groq.com/) (Llama-3.1-8B-Instant)
- **Vector Database & Retrieval**: [Pinecone](https://www.pinecone.io/)
- **Embeddings**: [Hugging Face](https://huggingface.co/) (`sentence-transformers/all-mpnet-base-v2`)
- **PDF Processing**: [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/), [pdfplumber](https://github.com/jsvine/pdfplumber)
- **Web Search**: [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/)

## 📦 Project Structure

```
legal-assistant-ai/
├── app.py                 # Main Streamlit application
├── Dockerfile            # Container configuration for Hugging Face deployment
├── requirements.txt      # Python dependencies
├── config.py            # API keys and configuration settings
├── rag_pipeline.py      # Core RAG chain and agent logic
├── embed_chunks_pinecone.py  # PDF text extraction and Pinecone embedding functions
├── legal_ingest.py      # PDF text extraction utilities
├── .dockerignore        # Files to exclude from Docker build
└── README.md            # This file
```

## 🏃‍♂️ Running Locally

Follow these steps to set up and run the project on your local machine.

### Prerequisites

- Python 3.10 or higher
- A Pinecone API Key ([Get one here](https://www.pinecone.io/))
- A Groq API Key ([Get one here](https://console.groq.com/))

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://huggingface.co/spaces/your-username/legal-assistant-ai
    cd legal-assistant-ai
    ```

2.  **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**
    Create a `.env` file in the root directory and add your API keys:
    ```ini
    PINECONE_API_KEY=your_pinecone_api_key_here
    GROQ_API_KEY=your_groq_api_key_here
    ```

5.  **Run the application**
    ```bash
    streamlit run app.py
    ```
    The app will open in your browser at `http://localhost:8501`.

## 🗝️ Configuration

The main configuration parameters in `config.py` are:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL_NAME` | Hugging Face model for creating embeddings | `all-mpnet-base-v2` |
| `GROQ_MODEL_NAME` | The Groq model to use for generation | `llama-3.1-8b-instant` |
| `PINECONE_INDEX_NAME` | Name of your Pinecone index | `legal-assistant` |
| `TOP_K` | Number of document chunks to retrieve for context | `5` |

## 📖 How to Use

1.  **Open the App**: Launch the app locally or open your Hugging Face Space.
2.  **Initialize Knowledge Base**:
    - In the sidebar, choose to upload a single PDF or provide a path to a folder of PDFs.
    - Click the "Process" button to chunk, embed, and store the documents in Pinecone.
3.  **Ask Questions**:
    - Type your legal question in the chat input (e.g., "Summarize the Anil Kumar Yadav vs State of NCT Delhi case").
    - The assistant will search through your documents and provide an answer with citations.
4.  **Toggle Web Search**: Use the "Web Search" toggle to let the assistant search the internet for answers instead of your documents.

## 🔒 License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://huggingface.co/spaces/your-username/legal-assistant-ai/discussions).

## 📜 Disclaimer

This tool is intended for research and informational purposes only. It is not a substitute for professional legal advice from a qualified attorney. Always verify any critical information against original sources and consult with a legal professional for advice on specific legal matters.

---

**Happy Legal Research!** ⚖️🔍
