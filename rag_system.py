import os
import json
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# for document processing
import PyPDF2
from bs4 import BeautifulSoup
import markdown

# for vector operations
import numpy as np
import faiss

# for Ollama interaction
import requests

# --- 1. Dataclass setup ---

@dataclass
class Chunk:
    """A structured container for a piece of document text."""
    text: str
    source: str
    page_number: Optional[int] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

# --- 2. Ollama Client for Embeddings and Generation ---

class OllamaClient:
    """Handles communication with the local Ollama server."""
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.embedding_model = "nomic-embed-text"
        self.llm_model = "llama3.2"
        self._check_ollama_status()

    def _check_ollama_status(self):
        """Checks if the Ollama server is running."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            print(f"Ollama server is running at {self.host}")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama server at {self.host}. Please ensure Ollama is running.")
            print(f"Details: {e}")
            raise ConnectionError("Ollama server not reachable.")

    def get_embedding(self, text: str) -> np.ndarray:
        """Generates an embedding for a given text using the specified model."""
        url = f"{self.host}/api/embed"
        payload = {
            "model": self.embedding_model,
            "input": text
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return np.array(data["embeddings"][0], dtype="float32")
        except requests.exceptions.RequestException as e:
            print(f"Error getting embedding from Ollama: {e}")
            return np.array([])

    def generate_response(self, prompt: str) -> str:
        """Generates a response from the LLM model."""
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "Error: No response from LLM.")
        except requests.exceptions.RequestException as e:
            print(f"Error generating response from Ollama: {e}")
            return "Error: Could not connect to or get a response from the LLM."

# --- 3. Document Loaders ---

def load_pdf(file_path: Path) -> List[Chunk]:
    """Extracts text from a PDF file, preserving page numbers."""
    chunks = []
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    chunks.append(Chunk(
                        text=text,
                        source=file_path.name,
                        page_number=i + 1
                    ))
    except Exception as e:
        print(f"Error loading PDF {file_path.name}: {e}")
    return chunks

def load_markdown(file_path: Path) -> List[Chunk]:
    """Extracts text from a Markdown file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            md_text = file.read()
            # Convert markdown to plain text for chunking
            html = markdown.markdown(md_text)
            plain_text = ''.join(BeautifulSoup(html, features="html.parser").findAll(text=True))
            return [Chunk(text=plain_text, source=file_path.name)]
    except Exception as e:
        print(f"Error loading Markdown {file_path.name}: {e}")
        return []

def load_html(file_path: Path) -> List[Chunk]:
    """Extracts text from an HTML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            plain_text = soup.get_text()
            return [Chunk(text=plain_text, source=file_path.name)]
    except Exception as e:
        print(f"Error loading HTML {file_path.name}: {e}")
        return []

def load_documents(directory: Path) -> List[Chunk]:
    """Loads all supported documents from a directory."""
    all_chunks = []
    for file_path in directory.iterdir():
        if file_path.suffix.lower() == ".pdf":
            all_chunks.extend(load_pdf(file_path))
        elif file_path.suffix.lower() in (".md", ".markdown"):
            all_chunks.extend(load_markdown(file_path))
        elif file_path.suffix.lower() in (".html", ".htm"):
            all_chunks.extend(load_html(file_path))
        else:
            print(f"Skipping unsupported file: {file_path.name}")
    return all_chunks

# --- 4. Text Chunker ---

def chunk_text(chunks: List[Chunk], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Chunk]:
    """Splits large text chunks into smaller, overlapping chunks."""
    final_chunks = []
    for chunk in chunks:
        text = chunk.text
        start = 0
        while start < len(text):
            end = start + chunk_size
            sub_text = text[start:end]
            
            # Simple way to find a natural break (e.g., end of a sentence)
            if end < len(text):
                last_period = sub_text.rfind('.')
                if last_period > chunk_size - chunk_overlap:
                    end = start + last_period + 1
                    sub_text = text[start:end]

            final_chunks.append(Chunk(
                text=sub_text,
                source=chunk.source,
                page_number=chunk.page_number
            ))
            start += chunk_size - chunk_overlap
            
            # Prevent infinite loop if chunk_size <= chunk_overlap
            if chunk_size <= chunk_overlap and start < len(text):
                start = len(text) # Break out
                
    return final_chunks

# --- 5. Vector Database (FAISS) and Indexing ---

def create_index(chunks: List[Chunk], ollama_client: OllamaClient) -> Tuple[faiss.Index, List[Chunk]]:
    """Generates embeddings and creates a FAISS index."""
    print("Generating embeddings and creating FAISS index...")
    embeddings = []
    indexed_chunks = []
    
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)} from {chunk.source}...", end="\r")
        embedding = ollama_client.get_embedding(chunk.text)
        if embedding.size > 0:
            chunk.embedding = embedding
            embeddings.append(embedding)
            indexed_chunks.append(chunk)
    
    if not embeddings:
        raise ValueError("No embeddings could be generated. Check Ollama server and model.")

    embeddings_matrix = np.stack(embeddings)
    dimension = embeddings_matrix.shape[1]
    
    # Use IndexFlatL2 for simple cosine similarity (L2 normalized vectors)
    index = faiss.IndexFlatL2(dimension)
    faiss.normalize_L2(embeddings_matrix) # Normalize for cosine similarity
    index.add(embeddings_matrix)
    
    print(f"\nFAISS index created with {index.ntotal} vectors of dimension {dimension}.")
    return index, indexed_chunks

def save_index(index: faiss.Index, chunks: List[Chunk], index_path: Path, metadata_path: Path):
    """Saves the FAISS index and chunk metadata."""
    faiss.write_index(index, str(index_path))
    metadata = [chunk.__dict__ for chunk in chunks]
    # Remove the large embedding array before saving metadata
    for item in metadata:
        item.pop('embedding', None)
        # Convert numpy types to native Python types for JSON serialization
        if item.get('page_number') is not None:
             item['page_number'] = int(item['page_number'])
             
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Index saved to {index_path} and metadata to {metadata_path}")

def load_index(index_path: Path, metadata_path: Path) -> Tuple[faiss.Index, List[Chunk]]:
    """Loads the FAISS index and chunk metadata."""
    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    chunks = [Chunk(**item) for item in metadata]
    print(f"Index loaded with {index.ntotal} vectors.")
    return index, chunks

# --- 6. RAG System Orchestration ---

class OfflineRAG:
    def __init__(self, documents_dir: str = "documents", index_file: str = "faiss_index.bin", metadata_file: str = "chunks_metadata.json"):
        self.ollama_client = OllamaClient()
        self.documents_dir = Path(documents_dir)
        self.index_path = Path(index_file)
        self.metadata_path = Path(metadata_file)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
        
        self._load_or_create_index()

    def _load_or_create_index(self):
        """Loads the index if it exists, otherwise creates it."""
        if self.index_path.exists() and self.metadata_path.exists():
            print("Loading existing index...")
            self.index, self.chunks = load_index(self.index_path, self.metadata_path)
        else:
            print("Index not found. Creating new index...")
            raw_chunks = load_documents(self.documents_dir)
            if not raw_chunks:
                raise FileNotFoundError(f"No supported documents found in {self.documents_dir}")
                
            processed_chunks = chunk_text(raw_chunks)
            self.index, self.chunks = create_index(processed_chunks, self.ollama_client)
            save_index(self.index, self.chunks, self.index_path, self.metadata_path)

    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        """Retrieves the top k most relevant chunks for a query."""
        query_embedding = self.ollama_client.get_embedding(query)
        if query_embedding.size == 0:
            return []
            
        query_vector = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query_vector)
        
        # D is distances, I is indices
        D, I = self.index.search(query_vector, k)
        
        retrieved_chunks = []
        for i in I[0]:
            if i != -1: # -1 means no result found
                retrieved_chunks.append(self.chunks[i])
                
        return retrieved_chunks

    def generate_prompt(self, query: str, retrieved_chunks: List[Chunk]) -> str:
        """Constructs the prompt for the LLM with context and query."""
        context = "\n---\n".join([
            f"Source: {c.source}, Page: {c.page_number or 'N/A'}\nContent: {c.text}"
            for c in retrieved_chunks
        ])
        
        prompt = f"""
        You are an expert Q&A system. Use the following context to answer the user's question.
        If you cannot find the answer in the context, state that you don't have enough information.
        
        CONTEXT:
        {context}
        
        QUESTION: {query}
        
        Answer:
        """
        return prompt

    def query(self, query: str) -> str:
        """Performs the full RAG query process."""
        print(f"\nProcessing query: '{query}'")
        
        # 1. Retrieval
        retrieved_chunks = self.retrieve(query)
        if not retrieved_chunks:
            return "Could not retrieve relevant information. Please check your documents and Ollama connection."
            
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks.")
        
        # 2. Augmentation and Generation
        llm_prompt = self.generate_prompt(query, retrieved_chunks)
        response = self.ollama_client.generate_response(llm_prompt)
        
        # 3. Add citations
        citations = set()
        for chunk in retrieved_chunks:
            citation = f"[{chunk.source}, Page {chunk.page_number or 'N/A'}]"
            citations.add(citation)
            
        citation_text = "\n\nCitations:\n" + "\n".join(sorted(list(citations)))
        
        return response + citation_text

# --- 7. Main Execution Block ---

if __name__ == "__main__":
    try:
        # 1. Initialize the RAG system (will load or create the index)
        rag_system = OfflineRAG()
        
        # 2. Example Query
        # The example document is about FLoRA: Fused forward-backward adapters
        example_query = "What is FLoRA and what problem does it aim to solve?"
        
        final_answer = rag_system.query(example_query)
        
        print("\n--- FINAL ANSWER ---")
        print(final_answer)
        print("---------------------\n")
        
        # 3. Interactive loop
        print("Starting interactive RAG session. Type 'exit' or 'quit' to end.")
        while True:
            user_query = input("Your question: ")
            if user_query.lower() in ["exit", "quit"]:
                break
            
            final_answer = rag_system.query(user_query)
            print("\n--- ANSWER ---")
            print(final_answer)
            print("--------------\n")
            
    except ConnectionError:
        print("\n--- SETUP REQUIRED ---")
        print("The Ollama server is not running or the models are not downloaded.")
        print("Please follow the setup instructions in the README to start Ollama and pull the models.")
    except FileNotFoundError as e:
        print(f"\n--- ERROR ---")
        print(e)
        print("Please ensure you have a 'documents' folder with supported files (PDF, MD, HTML).")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
