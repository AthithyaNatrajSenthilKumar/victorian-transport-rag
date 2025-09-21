"""
Document loading utilities for various file formats
"""
import os
from typing import List, Dict, Any
import pdfplumber
import docx2txt
from langchain.schema import Document


class DocumentLoader:
    """Handles loading of PDF and DOCX documents"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        
    def load_pdf(self, file_path: str) -> List[Document]:
        """Extract text from PDF using pdfplumber"""
        documents = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                if full_text.strip():
                    documents.append(Document(
                        page_content=full_text.strip(),
                        metadata={
                            "source": os.path.basename(file_path),
                            "file_path": file_path,
                            "file_type": "pdf",
                            "total_pages": len(pdf.pages)
                        }
                    ))
        except Exception as e:
            print(f"Error loading PDF {file_path}: {str(e)}")
            
        return documents
    
    def load_docx(self, file_path: str) -> List[Document]:
        """Extract text from DOCX files"""
        documents = []
        
        try:
            text = docx2txt.process(file_path)
            if text.strip():
                documents.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source": os.path.basename(file_path),
                        "file_path": file_path,
                        "file_type": "docx"
                    }
                ))
        except Exception as e:
            print(f"Error loading DOCX {file_path}: {str(e)}")
            
        return documents
    
    def load_all_documents(self) -> List[Document]:
        """Load all PDF and DOCX files from the data folder"""
        documents = []
        
        if not os.path.exists(self.data_folder):
            print(f"Warning: Data folder '{self.data_folder}' does not exist")
            return documents
        
        supported_extensions = ['.pdf', '.docx']
        
        for filename in os.listdir(self.data_folder):
            file_path = os.path.join(self.data_folder, filename)
            
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename.lower())
                
                if ext == '.pdf':
                    print(f"Loading PDF: {filename}")
                    documents.extend(self.load_pdf(file_path))
                elif ext == '.docx':
                    print(f"Loading DOCX: {filename}")
                    documents.extend(self.load_docx(file_path))
                else:
                    print(f"Skipping unsupported file: {filename}")
        
        print(f"\nLoaded {len(documents)} documents from {self.data_folder}")
        return documents


def get_document_stats(documents: List[Document]) -> Dict[str, Any]:
    """Get basic statistics about loaded documents"""
    if not documents:
        return {"total_docs": 0}
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    file_types = {}
    
    for doc in documents:
        file_type = doc.metadata.get('file_type', 'unknown')
        file_types[file_type] = file_types.get(file_type, 0) + 1
    
    return {
        "total_docs": len(documents),
        "total_characters": total_chars,
        "avg_chars_per_doc": total_chars // len(documents),
        "file_types": file_types,
        "sources": [doc.metadata.get('source', 'unknown') for doc in documents]
    }