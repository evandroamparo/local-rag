# Offline Retrieval-Augmented Generation (RAG) System

This project implements a fully offline RAG system based on the architecture described in the HackerNoon article "Building a RAG System That Runs Completely Offline." It uses **Ollama** for local Large Language Model (LLM) and embedding generation, **FAISS** for the vector database, and standard Python libraries for document processing.

The entire system, once models are downloaded, runs without any internet connection, ensuring maximum data privacy and security.

## Prerequisites

You will need the following installed on your system:

1.  **Python 3.8+**: For running the RAG script.
2.  **Ollama**: The tool used to run open-source LLMs and embedding models locally.

## Step 1: Ollama Setup

Ollama is essential for running the models locally. You must install it and download the required models before running the Python script.

### 1. Install Ollama

Download and install Ollama for your operating system (Windows, macOS, or Linux) from the official website:

> **[https://ollama.com/download](https://ollama.com/download)**

For Linux users, you can typically run:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Download Models

Once Ollama is installed, you need to pull the LLM and the embedding model used in the project.

**Start the Ollama server (if it's not running as a service):**

```bash
ollama serve &
```

**Pull the LLM (Llama 3.2 - ~2GB):**

```bash
ollama pull llama3.2
```

**Pull the Embedding Model (nomic-embed-text - ~274MB):**

```bash
ollama pull nomic-embed-text
```

## Step 2: Project Setup

### 1. Install Python Dependencies

It is highly recommended to use a Python virtual environment.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows

# Install the required packages
pip install -r requirements.txt
```

The required packages are listed in `requirements.txt`:
*   `faiss-cpu`: The fast vector search library.
*   `numpy`: For array operations.
*   `PyPDF2`: For PDF text extraction.
*   `beautifulsoup4` and `markdown`: For processing HTML and Markdown documents.
*   `requests`: For communicating with the local Ollama server.

### 2. Add Your Documents

The RAG system is configured to read documents from a folder named `documents/`.

*   Create a folder named `documents` in the project root.
*   Place your PDF, Markdown (`.md`), or HTML (`.html`) files inside this folder.

> **Note:** An example PDF (`FLoRA.pdf`) has been included in the `documents/` folder for initial testing.

## Step 3: Run the RAG System

Ensure the Ollama server is running (see Step 1.2). Then, run the main Python script:

```bash
python rag_system.py
```

### First Run (Indexing)

On the first run, the script will:
1.  Load documents from the `documents/` folder.
2.  Chunk the text.
3.  Generate embeddings for all chunks using `nomic-embed-text`.
4.  Create a FAISS index and save it to `faiss_index.bin` and `chunks_metadata.json`.

This process can take a few minutes depending on the size and number of your documents.

### Subsequent Runs

In subsequent runs, the script will automatically load the saved index, making the startup process much faster.

### Usage

After the initial indexing, the script will automatically run an example query and then enter an interactive loop where you can ask questions about your documents.

```
Your question: What is FLoRA and what problem does it aim to solve?
```

Type `exit` or `quit` to end the interactive session.

## Project Files

| File Name | Description |
| :--- | :--- |
| `rag_system.py` | The core Python script implementing the RAG logic (document loading, chunking, embedding, indexing, retrieval, and generation). |
| `requirements.txt` | List of Python dependencies. |
| `README.md` | This instruction file. |
| `documents/` | Directory where your source documents (PDF, MD, HTML) should be placed. |
| `documents/FLoRA.pdf` | Example PDF document for testing. |
| `faiss_index.bin` | (Generated on first run) The FAISS vector index. |
| `chunks_metadata.json` | (Generated on first run) Metadata for the chunks corresponding to the index. |
