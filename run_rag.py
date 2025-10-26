"""
Main RAG Pipeline for Victorian Public Transport FAQ System
(works with split packages: langchain-core/community/huggingface/ollama)
"""

import os
import sys
from typing import List, Dict, Any, Tuple
import warnings

warnings.filterwarnings("ignore")

# utils path
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

# Vector store + embeddings + LLM (split packages)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Core prompt
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Lightweight cross-encoder for reranking
from sentence_transformers import CrossEncoder

from loaders import DocumentLoader, get_document_stats
from chunkers import SmartChunker


class VictorianTransportRAG:
    """RAG system for Victorian Public Transport FAQ"""

    def __init__(
        self,
        data_folder: str = "data",
        model_name: str = "llama3",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        k: int = 6,  # final number of docs after rerank
    ):
        self.data_folder = data_folder
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.device = device
        self.k = k

        self.vectorstore = None
        self.retriever = None
        self.prompt: PromptTemplate | None = None

        # Keep processed docs/chunks for health checks and debugging
        self.documents: List[Document] = []
        self.chunks: List[Document] = []

        # Components
        self.loader = DocumentLoader(data_folder)
        self.chunker = SmartChunker(chunk_size=800, chunk_overlap=200)

        # Embeddings (cosine-normalized)
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

        # LLM (local via Ollama)
        print(f"Connecting to Ollama model: {model_name}")
        self.llm = OllamaLLM(model=model_name, temperature=0.1)

        # Small CPU-friendly reranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.rerank_top_k = self.k
        self.fetch_k = 30  # how many to retrieve before reranking

    # --------------------------------------------------------------------- #
    # Build index
    # --------------------------------------------------------------------- #
    def load_and_process_documents(self) -> bool:
        """Load, chunk and index all documents into FAISS."""
        print("=== Loading Documents ===")
        self.documents = self.loader.load_all_documents()

        if not self.documents:
            print("No documents found. Please add PDF or DOCX files to the data folder.")
            return False

        stats = get_document_stats(self.documents)
        print(f"Document Stats: {stats}")

        print("\n=== Chunking Documents ===")
        raw_chunks = self.chunker.chunk_with_context(self.documents)

        # Normalize metadata, drop short/noisy chunks
        self.chunks = []
        for idx, d in enumerate(raw_chunks):
            text = (d.page_content or "").strip()
            if len(text) < 150:
                continue
            if "chunk_id" not in d.metadata:
                d.metadata["chunk_id"] = idx
            if "source" not in d.metadata:
                d.metadata["source"] = (
                    d.metadata.get("file_path")
                    or d.metadata.get("path")
                    or d.metadata.get("document_id")
                    or "unknown"
                )
            self.chunks.append(d)

        chunk_stats = self.chunker.get_chunk_stats(self.chunks)
        print(f"Chunk Stats: {chunk_stats}")

        print("\n=== Creating Vector Store ===")
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
        print("Vector store created successfully!")
        return True

    # --------------------------------------------------------------------- #
    # Prompt + retriever setup
    # --------------------------------------------------------------------- #
    def setup_qa_chain(self):
        """Configure retriever and prompt. No dependency on monolithic langchain."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Run load_and_process_documents() first.")

        self.prompt = PromptTemplate(
            template=(
                "You are a helpful assistant for Victorian public transport information.\n"
                "Use ONLY the provided context to answer questions about fares, zones, accessibility, "
                "penalties, and travel rules. If the answer is not in the context, say you don't have "
                "that specific information.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )

        # MMR retriever for diversity
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "fetch_k": self.fetch_k, "lambda_mult": 0.5},
        )
        print("Retriever and prompt set up successfully!")

    # --------------------------------------------------------------------- #
    # Inference
    # --------------------------------------------------------------------- #
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Retrieve, rerank, stuff the context, and call the LLM."""
        if self.retriever is None or self.prompt is None:
            raise ValueError("QA not set up. Run setup_qa_chain() first.")

        print(f"\nQuestion: {question}")

        # 1) Retrieve candidates (modern API)
        candidates: List[Document] = self.retriever.invoke(question)

        # 2) Rerank with cross-encoder
        try:
            pairs = [(question, d.page_content) for d in candidates]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            top_docs = [d for d, _ in ranked[: self.rerank_top_k]]
        except Exception:
            top_docs = candidates[: self.k]

        # 3) Build stuffed context
        def _one(doc: Document) -> str:
            meta = doc.metadata or {}
            src = meta.get("source", meta.get("file_path", "unknown"))
            txt = (doc.page_content or "").strip()
            return f"[Source: {src} | Chunk: {meta.get('chunk_id', 0)}]\n{txt}"

        context = "\n\n---\n\n".join(_one(d) for d in top_docs)

        # 4) Prompt and invoke LLM
        filled = self.prompt.format(context=context, question=question)
        answer = self.llm.invoke(filled)

        # 5) Format sources
        def _fmt(doc: Document) -> Dict[str, Any]:
            preview = (doc.page_content or "").strip()
            if len(preview) > 200:
                preview = preview[:200].rstrip() + "..."
            meta = doc.metadata or {}
            return {
                "content": preview,
                "source": meta.get("source", meta.get("file_path", "unknown")),
                "chunk_id": int(meta.get("chunk_id", 0)),
            }

        return {
            "question": question,
            "answer": answer,
            "source_documents": [_fmt(doc) for doc in top_docs],
        }

    # --------------------------------------------------------------------- #
    # Health / diagnostics
    # --------------------------------------------------------------------- #
    def index_health(self) -> Tuple[int | None, int | None]:
        """Returns (n_chunks, n_vectors) for a quick consistency check."""
        try:
            n_chunks = len(self.chunks)
        except Exception:
            n_chunks = None

        n_vectors = None
        try:
            if hasattr(self, "vectorstore") and hasattr(self.vectorstore, "index"):
                n_vectors = int(getattr(self.vectorstore.index, "ntotal", 0))
        except Exception:
            pass

        return n_chunks, n_vectors

    # --------------------------------------------------------------------- #
    # CLI helpers
    # --------------------------------------------------------------------- #
    def print_response(self, response: Dict[str, Any]):
        print("\n" + "=" * 50)
        print(f"Q: {response['question']}")
        print(f"\nA: {response['answer']}")
        print("\nSources:")
        for i, source in enumerate(response["source_documents"], 1):
            print(f"{i}. {source['source']} (chunk {source['chunk_id']})")
            print(f"   Preview: {source['content']}")
        print("=" * 50)


def main():
    print("Victorian Public Transport RAG System")
    print("====================================")

    rag = VictorianTransportRAG()

    if not rag.load_and_process_documents():
        return

    rag.setup_qa_chain()

    # Health check
    n_chunks, n_vectors = rag.index_health()
    if n_chunks is not None and n_vectors is not None:
        status = "OK" if n_chunks == n_vectors else "MISMATCH"
        print(f"\nIndex health: chunks={n_chunks}, vectors={n_vectors}  [{status}]")

    # Sample FAQ questions for testing
    sample_questions = [
        "What are the penalty fares for travelling without a valid ticket?",
        "How much does a daily Zone 1+2 myki card cost?",
        "What accessibility features are available on public transport?",
        "Can I use my myki card on regional trains?",
        "What are the peak and off-peak travel times?",
    ]

    print("\n=== Testing with Sample Questions ===")
    for question in sample_questions:
        try:
            response = rag.ask_question(question)
            rag.print_response(response)
        except Exception as e:
            print(f"Error processing question '{question}': {str(e)}")

    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Type 'quit' to exit")
    while True:
        try:
            question = input("\nEnter your question: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                break
            if not question:
                continue
            response = rag.ask_question(question)
            rag.print_response(response)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
