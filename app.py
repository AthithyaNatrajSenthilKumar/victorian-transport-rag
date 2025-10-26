"""
Streamlit Web UI for Victorian Public Transport RAG System

This application provides a web interface for querying information about Victorian public transport
using Retrieval-Augmented Generation (RAG). It combines document retrieval with language models
to provide accurate, source-backed answers to user queries.
"""

import os
import sys
import time
from typing import Dict, Any

import streamlit as st

print("Initializing Victorian Transport RAG System...")

# ---------------- Session state ----------------
print("Setting up session state...")
if "history" not in st.session_state:
    # [{"q": str, "answer": str, "sources": list, "latency_ms": float}]
    st.session_state.history = []

# utils path
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from run_rag import VictorianTransportRAG  # noqa: E402


# ---------------- RAG init ----------------
@st.cache_resource
def initialize_rag_system():
    """
    Initialize the RAG system with document processing and QA chain setup.
    
    Returns:
        VictorianTransportRAG: Initialized RAG system instance if successful, None otherwise.
    """
    print("Initializing RAG system...")
    rag = VictorianTransportRAG()
    
    print("Loading and processing documents...")
    if rag.load_and_process_documents():
        print("Setting up QA chain...")
        rag.setup_qa_chain()
        print("RAG system initialization complete!")
        return rag
    
    print("Failed to initialize RAG system - document loading failed")
    return None


# ---------------- Helpers ----------------
def format_sources(sources) -> str:
    """
    Format source documents for display in the web interface.
    
    Args:
        sources: List of source documents, either as dicts or LangChain Document objects
        
    Returns:
        str: Formatted string containing source information and content previews
        
    Note:
        - Handles both dictionary and LangChain Document formats
        - Truncates previews to 600 characters for readability
        - Includes source name, part number, and content preview
    """
    print(f"Formatting {len(sources)} source documents for display...")
    formatted = []
    for i, s in enumerate(sources, 1):
        if hasattr(s, "page_content"):  # Document
            content = getattr(s, "page_content", "")
            meta = getattr(s, "metadata", {}) or {}
            source_name = meta.get("source") or meta.get("file_path") or meta.get("path") or "Unknown source"
            chunk_idx = meta.get("chunk_id")
        else:  # dict
            content = s.get("content") or s.get("page_content") or ""
            meta = s.get("metadata", {}) or {}
            source_name = s.get("source") or meta.get("source") or meta.get("file_path") or "Unknown source"
            chunk_idx = s.get("chunk_id") or meta.get("chunk_id")

        part_txt = f"(Part {int(chunk_idx) + 1})" if chunk_idx is not None else ""
        formatted.append(f"**Source {i}**: {source_name} {part_txt}".strip())

        preview = (content or "").strip()
        if len(preview) > 600:
            preview = preview[:600].rstrip() + " …"
        formatted.append(f"*Preview*: {preview}")
        formatted.append("---")
    return "\n\n".join(formatted)


# ---------------- App ----------------
def main():
    """
    Main application function that sets up the Streamlit interface and handles user interactions.
    
    This function:
    - Configures the page layout and appearance
    - Initializes the RAG system
    - Manages the conversation history
    - Handles user input and question processing
    - Displays results and source documents
    """
    print("Starting main application...")
    
    st.set_page_config(
        page_title="Victorian Transport FAQ Assistant",
        page_icon="🚊",
        layout="wide",
    )

    st.title("🚊 Victorian Public Transport FAQ Assistant")
    st.markdown("*Powered by RAG (Retrieval-Augmented Generation)*")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write(
            """
            This AI assistant helps answer questions about Victorian public transport using official documents.

            **Features:**
            - 🎫 Fare information
            - 🗺️ Zone details
            - ♿ Accessibility info
            - 📋 Travel rules & penalties
            - 🚌 Service information
            """
        )

        st.header("Session")
        if st.button("🧹 Clear conversation"):
            st.session_state.history = []
            st.rerun()

        st.header("Sample Questions")
        sample_questions = [
            "What are the penalty fares?",
            "How much is a Zone 1+2 daily ticket?",
            "What accessibility features are available?",
            "Can I use myki on regional trains?",
            "What are peak travel times?",
        ]
        for q in sample_questions:
            if st.button(f"💭 {q}", key=f"sample_{hash(q)}"):
                st.session_state.sample_question = q

    # Init RAG
    with st.spinner("Loading knowledge base..."):
        rag_system = initialize_rag_system()

    if rag_system is None:
        st.error("❌ Failed to load documents. Please ensure PDF/DOCX files are in the 'data' folder.")
        st.info("📁 Expected folder structure: `data/` containing your transport documents")
        return

    st.success("✅ Knowledge base loaded successfully!")

    # Conversation history
    if st.session_state.history:
        st.subheader("🧠 Conversation")
        for i, turn in enumerate(st.session_state.history, start=1):
            st.markdown(f"**Q{i}.** {turn['q']}")
            st.write(turn["answer"])
            if "latency_ms" in turn:
                st.caption(f"⏱ {turn['latency_ms']:.0f} ms")
            with st.expander(f"📚 View Sources for Q{i}", expanded=False):
                if turn.get("sources"):
                    st.markdown("**Retrieved Information:**")
                    st.markdown(format_sources(turn["sources"]))
                else:
                    st.write("No source documents retrieved.")
            st.markdown("---")

    # Ask next question (bottom)
    st.markdown("### Ask your question about Victorian public transport:")

    with st.form("ask_form", clear_on_submit=True):
        q = st.text_input(
            label="",
            placeholder="e.g., What are the current myki fares?",
            value=st.session_state.get("sample_question", ""),
        )
        submit = st.form_submit_button(
            "🔎 Ask Question", use_container_width=True, key="ask_btn_bottom"
        )

    # Clear the sample question only after successful use
    if "sample_question" in st.session_state and submit and q.strip():
        del st.session_state.sample_question

    # Process question
    if submit:
        if not q.strip():
            st.warning("⚠️ Please enter a question first.")
        else:
            with st.spinner("Searching for answer..."):
                try:
                    print(f"Processing question: {q.strip()}")
                    t0 = time.perf_counter()
                    response = rag_system.ask_question(q.strip())
                    t1 = time.perf_counter()
                    latency = (t1 - t0) * 1000
                    
                    print(f"Question processed successfully in {latency:.2f}ms")
                    print(f"Retrieved {len(response.get('source_documents', []))} source documents")

                    st.session_state.history.append(
                        {
                            "q": q.strip(),
                            "answer": response.get("answer", ""),
                            "sources": response.get("source_documents", []),
                            "latency_ms": latency,
                        }
                    )
                    st.rerun()
                except Exception as e:
                    print(f"Error occurred while processing question: {str(e)}")
                    st.error(f"❌ Error processing question: {str(e)}")
                    st.info("Please try rephrasing your question or check if the Ollama service is running.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <small>
            Built with Streamlit, LangChain, and Ollama |
            Using sentence-transformers for embeddings |
            FAISS for vector search
            </small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    print("=" * 50)
    print("Victorian Transport RAG System - Application Start")
    print("=" * 50)
    main()
    print("Application terminated")
