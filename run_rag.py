"""
Main RAG Pipeline for Victorian Public Transport FAQ System
"""
import os
import sys
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from loaders import DocumentLoader, get_document_stats
from chunkers import SmartChunker


class VictorianTransportRAG:
    """RAG system for Victorian Public Transport FAQ"""
    
    def __init__(self, data_folder: str = "data", model_name: str = "llama3"):
        self.data_folder = data_folder
        self.model_name = model_name
        self.vectorstore = None
        self.qa_chain = None
        
        # Initialize components
        self.loader = DocumentLoader(data_folder)
        self.chunker = SmartChunker(chunk_size=800, chunk_overlap=150)
        
        # Initialize embeddings (free HuggingFace model)
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize LLM
        print(f"Connecting to Ollama model: {model_name}")
        self.llm = Ollama(model=model_name, temperature=0.1)
        
    def load_and_process_documents(self):
        """Load and process all documents"""
        print("=== Loading Documents ===")
        documents = self.loader.load_all_documents()
        
        if not documents:
            print("No documents found. Please add PDF or DOCX files to the data folder.")
            return False
        
        # Print document stats
        stats = get_document_stats(documents)
        print(f"Document Stats: {stats}")
        
        print("\n=== Chunking Documents ===")
        chunks = self.chunker.chunk_with_context(documents)
        
        # Print chunk stats
        chunk_stats = self.chunker.get_chunk_stats(chunks)
        print(f"Chunk Stats: {chunk_stats}")
        
        print("\n=== Creating Vector Store ===")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("Vector store created successfully!")
        
        return True
    
    def setup_qa_chain(self):
        """Set up the QA chain with custom prompt"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Run load_and_process_documents() first.")
        
        # Custom prompt template for Victorian transport queries
        prompt_template = """You are a helpful assistant for Victorian public transport information. 
Use the provided context to answer questions about fares, zones, accessibility, penalties, and travel rules.
If the answer is not in the context, politely say you don't have that specific information.

Context: {context}

Question: {question}

Answer: """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        print("QA chain set up successfully!")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get answer with sources"""
        if not self.qa_chain:
            raise ValueError("QA chain not set up. Run setup_qa_chain() first.")
        
        print(f"\nQuestion: {question}")
        
        # Get answer
        result = self.qa_chain({"query": question})
        
        # Format response
        response = {
            "question": question,
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content[:200] + "...",  # First 200 chars
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0)
                }
                for doc in result["source_documents"]
            ]
        }
        
        return response
    
    def print_response(self, response: Dict[str, Any]):
        """Pretty print the response"""
        print(f"\n{'='*50}")
        print(f"Q: {response['question']}")
        print(f"\nA: {response['answer']}")
        print(f"\nSources:")
        for i, source in enumerate(response['source_documents'], 1):
            print(f"{i}. {source['source']} (chunk {source['chunk_id']})")
            print(f"   Preview: {source['content']}")
        print("="*50)


def main():
    """Main execution function"""
    print("Victorian Public Transport RAG System")
    print("====================================")
    
    # Initialize RAG system
    rag = VictorianTransportRAG()
    
    # Load and process documents
    if not rag.load_and_process_documents():
        return
    
    # Set up QA chain
    rag.setup_qa_chain()
    
    # Sample FAQ questions for testing
    sample_questions = [
        "What are the penalty fares for travelling without a valid ticket?",
        "How much does a daily Zone 1+2 myki card cost?",
        "What accessibility features are available on public transport?",
        "Can I use my myki card on regional trains?",
        "What are the peak and off-peak travel times?"
    ]
    
    print(f"\n=== Testing with Sample Questions ===")
    
    for question in sample_questions:
        try:
            response = rag.ask_question(question)
            rag.print_response(response)
        except Exception as e:
            print(f"Error processing question '{question}': {str(e)}")
    
    # Interactive mode
    print(f"\n=== Interactive Mode ===")
    print("Type 'quit' to exit")
    
    while True:
        try:
            question = input("\nEnter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
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