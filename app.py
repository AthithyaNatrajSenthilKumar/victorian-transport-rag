"""
Streamlit Web UI for Victorian Public Transport RAG System
"""
import streamlit as st
import os
import sys
from typing import Dict, Any

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from run_rag import VictorianTransportRAG


@st.cache_resource
def initialize_rag_system():
    """Initialize and cache the RAG system"""
    rag = VictorianTransportRAG()
    
    # Load and process documents
    if rag.load_and_process_documents():
        rag.setup_qa_chain()
        return rag
    else:
        return None


def format_sources(sources):
    """Format source documents for display"""
    formatted = []
    for i, source in enumerate(sources, 1):
        formatted.append(f"**Source {i}**: {source['source']} (Part {source['chunk_id'] + 1})")
        formatted.append(f"*Preview*: {source['content']}")
        formatted.append("---")
    return "\n\n".join(formatted)


def main():
    st.set_page_config(
        page_title="Victorian Transport FAQ Assistant",
        page_icon="🚊",
        layout="wide"
    )
    
    # Header
    st.title("🚊 Victorian Public Transport FAQ Assistant")
    st.markdown("*Powered by RAG (Retrieval-Augmented Generation)*")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.write("""
        This AI assistant helps answer questions about Victorian public transport using official documents.
        
        **Features:**
        - 🎫 Fare information
        - 🗺️ Zone details  
        - ♿ Accessibility info
        - 📋 Travel rules & penalties
        - 🚌 Service information
        """)
        
        st.header("Sample Questions")
        sample_questions = [
            "What are the penalty fares?",
            "How much is a Zone 1+2 daily ticket?", 
            "What accessibility features are available?",
            "Can I use myki on regional trains?",
            "What are peak travel times?"
        ]
        
        for q in sample_questions:
            if st.button(f"💭 {q}", key=f"sample_{hash(q)}"):
                st.session_state.sample_question = q
    
    # Initialize RAG system
    with st.spinner("Loading knowledge base..."):
        rag_system = initialize_rag_system()
    
    if rag_system is None:
        st.error("❌ Failed to load documents. Please ensure PDF/DOCX files are in the 'data' folder.")
        st.info("📁 Expected folder structure: `data/` containing your transport documents")
        return
    
    st.success("✅ Knowledge base loaded successfully!")
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Question input
        question = st.text_input(
            "Ask your question about Victorian public transport:",
            placeholder="e.g., What are the current myki fares?",
            value=st.session_state.get('sample_question', '')
        )
        
        # Clear sample question after use
        if 'sample_question' in st.session_state:
            del st.session_state.sample_question
    
    with col2:
        ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
    
    # Process question
    if ask_button and question.strip():
        with st.spinner("Searching for answer..."):
            try:
                response = rag_system.ask_question(question.strip())
                
                # Display answer
                st.subheader("📝 Answer")
                st.write(response['answer'])
                
                # Display sources in expandable section
                with st.expander("📚 View Sources", expanded=False):
                    if response['source_documents']:
                        st.markdown("**Retrieved Information:**")
                        sources_text = format_sources(response['source_documents'])
                        st.markdown(sources_text)
                    else:
                        st.write("No source documents retrieved.")
                
                # Feedback section
                st.subheader("📊 Was this helpful?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("👍 Yes", key="feedback_yes"):
                        st.success("Thank you for your feedback!")
                with col_no:
                    if st.button("👎 No", key="feedback_no"):
                        st.info("We'll work on improving our responses!")
                        
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                st.info("Please try rephrasing your question or check if the Ollama service is running.")
    
    elif ask_button:
        st.warning("⚠️ Please enter a question first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>
        Built with Streamlit, LangChain, and Ollama | 
        Using sentence-transformers for embeddings | 
        FAISS for vector search
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()