import tiktoken
from pathlib import Path
from rag_system import load_documents, chunk_text

def estimate_costs(num_pages: int, avg_tokens_per_chunk: float, chunks_per_page: float, embedding_model_cost: float):
    """
    Estimate costs for indexing pages using OpenAI models.

    Args:
        num_pages: Number of pages to index
        avg_tokens_per_chunk: Average tokens per chunk
        chunks_per_page: Average chunks per page
        embedding_model_cost: Cost per 1M tokens for embeddings

    Returns:
        dict with costs
    """
    total_chunks = num_pages * chunks_per_page
    total_tokens_embeddings = total_chunks * avg_tokens_per_chunk

    embedding_cost_dollars = (total_tokens_embeddings / 1_000_000) * embedding_model_cost

    # During indexing, no LLM calls typically, just embeddings

    return {
        "total_embedding_tokens": total_tokens_embeddings,
        "embedding_cost_usd": embedding_cost_dollars,
        "total_cost_usd": embedding_cost_dollars  # Assuming only embeddings for indexing
    }

if __name__ == "__main__":
    # Load existing documents and calculate actual stats
    documents_dir = Path("documents")
    raw_chunks = load_documents(documents_dir)
    print(f"Total raw pages/chunks: {len(raw_chunks)}")

    # Calculate total pages (assuming PDFs have page per chunk)
    total_pages = len([c for c in raw_chunks if c.page_number])
    print(f"Total PDF pages: {total_pages}")

    processed_chunks = chunk_text(raw_chunks)
    print(f"Total processed chunks: {len(processed_chunks)}")

    # Calculate tokens for all chunk texts
    enc = tiktoken.get_encoding("cl100k_base")  # For text-embedding-3-small, which uses cl100k

    total_tokens = 0
    for chunk in processed_chunks:
        tokens = len(enc.encode(chunk.text))
        total_tokens += tokens

    avg_tokens_per_chunk = total_tokens / len(processed_chunks) if processed_chunks else 0
    chunks_per_page = len(processed_chunks) / total_pages if total_pages > 0 else 0

    print(f"Total chunks: {len(processed_chunks)}")
    print(f"Total tokens for embeddings: {total_tokens:,}")
    print(f"Average tokens per chunk: {avg_tokens_per_chunk:.2f}")
    print(f"Chunks per page: {chunks_per_page:.2f}")

    # Costs
    embedding_cost_per_1m = 0.020  # text-embedding-3-small
    llm_input_per_1m = 0.150  # gpt-4o-mini input
    llm_output_per_1m = 0.600  # gpt-4o-mini output

    # For a sample of 100 pages
    sample_pages = 100
    estimate = estimate_costs(sample_pages, avg_tokens_per_chunk, chunks_per_page, embedding_cost_per_1m)
    print(f"\nFor {sample_pages} pages:")
    print(f"Estimated embedding tokens: {estimate['total_embedding_tokens']:,}")
    print(f"Estimated embedding cost: ${estimate['embedding_cost_usd']:.6f}")

    # Also assume query costs
    # Query: 1 embedding + 1 LLM call (assume context 1000 tokens input, 200 output)
    query_input_tokens = 20 + 1000  # query + context
    query_output_tokens = 200
    query_cost = (query_input_tokens / 1_000_000 * llm_input_per_1m) + (query_output_tokens / 1_000_000 * llm_output_per_1m)
    print(f"Estimated cost per query: ${query_cost:.6f}")

    # Provide the function for general estimation
    print("\nUse estimate_costs(num_pages, avg_tokens_per_chunk, chunks_per_page, embedding_cost_per_1m) to estimate for other values.")
    print(f"Example: estimate_costs({sample_pages}, {avg_tokens_per_chunk:.2f}, {chunks_per_page:.2f}, {embedding_cost_per_1m}) -> ${estimate['embedding_cost_usd']:.6f}")
