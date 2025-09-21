"""
Document chunking utilities for RAG pipeline
"""
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentChunker:
    """Handles document chunking with various strategies"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            return []
        
        print(f"Chunking {len(documents)} documents...")
        print(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        chunked_docs = []
        
        for doc in documents:
            # Split the document
            chunks = self.text_splitter.split_text(doc.page_content)
            
            # Create new Document objects for each chunk
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,  # Inherit original metadata
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk)
                    }
                )
                chunked_docs.append(chunk_doc)
        
        print(f"Created {len(chunked_docs)} chunks")
        return chunked_docs
    
    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """Get statistics about chunks"""
        if not chunks:
            return {"total_chunks": 0}
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) // len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_characters": sum(chunk_sizes)
        }


class SmartChunker(DocumentChunker):
    """Enhanced chunker that preserves document structure"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        
        # Custom splitter that respects document structure
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n",  # Multiple line breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence endings
                "! ",
                "? ",
                "; ",      # Clause breaks
                ", ",      # Comma breaks
                " ",       # Word breaks
                ""         # Character breaks (last resort)
            ],
            length_function=len
        )
    
    def chunk_with_context(self, documents: List[Document]) -> List[Document]:
        """Chunk documents while preserving context information"""
        chunked_docs = self.chunk_documents(documents)
        
        # Add context information to each chunk
        for chunk in chunked_docs:
            source = chunk.metadata.get('source', 'unknown')
            chunk_id = chunk.metadata.get('chunk_id', 0)
            
            # Add context prefix to chunk content
            context_prefix = f"[Source: {source}, Part {chunk_id + 1}]\n"
            chunk.page_content = context_prefix + chunk.page_content
        
        return chunked_docs