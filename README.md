# Victorian Public Transport RAG System 🚊

A Retrieval-Augmented Generation (RAG) system for answering questions about Victorian public transport using official documents. Built for RMIT WIL project following test-driven RAG methodology.

## 🎯 Project Overview

This RAG system demonstrates how AI can add value to customer service by:
- Answering FAQ questions using official transport documents
- Providing source citations for transparency
- Reducing response time compared to manual document lookup
- Offering consistent, accurate information 24/7

**Built with:** LangChain, FAISS, Ollama, Streamlit, HuggingFace Transformers

---

## 🚀 Complete Setup Guide

### Prerequisites
- Python 3.8+ (tested on Python 3.11)
- Windows/macOS/Linux
- At least 8GB RAM recommended
- Internet connection for initial model downloads

### Step 1: Clone/Download Project
```bash
# Download project files to your desired location
# Ensure you have this folder structure:
rag-chatbot/
├── run_rag.py
├── app.py  
├── evaluate.py
├── requirements.txt
├── README.md
├── data/              # Create this folder
├── utils/             # Create this folder
│   ├── __init__.py
│   ├── loaders.py
│   └── chunkers.py
└── examples/
    └── faq_samples.json
```

### Step 2: Create Python Virtual Environment
```bash
# Navigate to project folder
cd path/to/rag-chatbot

# Create virtual environment
python -m venv rag-env

# Activate environment
# Windows:
rag-env\Scripts\activate
# macOS/Linux:
source rag-env/bin/activate

# You should see (rag-env) in your terminal prompt
```

### Step 3: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install all required packages
pip install -r requirements.txt

# This will install:
# - LangChain ecosystem for RAG
# - FAISS for vector search
# - Sentence Transformers for embeddings
# - Document processing libraries
# - Streamlit for web interface
# - Ollama for local LLM
```

### Step 4: Install and Setup Ollama

**Windows:**
1. Download Ollama from https://ollama.com/
2. Run the installer
3. Open Command Prompt and run:
```bash
ollama pull llama3
```

**macOS:**
```bash
# Install via Homebrew
brew install ollama

# Pull the model
ollama pull llama3
```

**Linux:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3
```

**Verify Installation:**
```bash
ollama --version
ollama list  # Should show llama3
```

### Step 5: Prepare Your Documents
```bash
# Create data folder if not exists
mkdir data

# Add your transport documents:
# - Copy PDF files (fares guides, accessibility info, etc.)
# - Copy DOCX files (conditions of travel, policies, etc.)
# - Supported formats: .pdf, .docx
```

**Example data folder:**
```
data/
├── transport_fares_2024.pdf
├── accessibility_guide.pdf  
├── zone_maps.pdf
├── conditions_of_travel.docx
├── penalty_fares.pdf
└── myki_user_guide.pdf
```

### Step 6: Start Ollama Server
```bash
# Start Ollama server (keep this running)
ollama serve

# You should see output like:
# "Listening on 127.0.0.1:11434"
# Keep this terminal open!
```

### Step 7: Run the RAG System

Open a **NEW TERMINAL** and activate your environment:
```bash
cd path/to/rag-chatbot
rag-env\Scripts\activate  # Windows
# source rag-env/bin/activate  # macOS/Linux
```

**Option A: Web Interface (Recommended)**
```bash
streamlit run app.py
```
- Opens in browser at `http://localhost:8501`
- User-friendly interface with sample questions
- Source document viewing
- Perfect for demonstrations

**Option B: Command Line Interface**
```bash
python run_rag.py
```
- Terminal-based interaction
- Good for testing and development
- Shows detailed technical information

**Option C: Run Evaluation**
```bash
python evaluate.py
```
- Tests system performance on 15 sample questions
- Generates metrics and CSV results
- Essential for measuring system effectiveness

---

## 📊 Understanding the Output

### Web Interface Features:
- **Question Input**: Type your transport-related questions
- **Sample Questions**: Pre-loaded FAQ examples
- **Answer Display**: AI-generated responses
- **Source Citations**: View retrieved document chunks
- **Feedback**: Rate answer quality

### Command Line Features:
- **Document Loading**: See processing statistics
- **Chunking Info**: View how documents are split
- **Interactive Q&A**: Type questions and get immediate responses
- **Source Details**: Full context of retrieved information

### Evaluation Metrics:
- **Answer Coverage**: % of questions receiving responses
- **Confidence Rate**: % of confident (non-"don't know") answers
- **Response Time**: Average query processing speed
- **Source Quality**: Retrieval accuracy metrics
- **Category Performance**: Success rates by question type

---

## 🔧 Troubleshooting

### Common Issues:

**❌ "No documents found"**
- Solution: Add PDF/DOCX files to the `data/` folder
- Check file permissions and formats

**❌ "Ollama connection failed"**
- Solution: Ensure Ollama server is running (`ollama serve`)
- Check if `ollama list` shows llama3 model
- Try restarting Ollama service

**❌ "Module not found" errors**
- Solution: Ensure virtual environment is activated
- Re-run `pip install -r requirements.txt`
- Check all files are in correct folders

**❌ Memory/Performance issues**
- Solution: Reduce document size or quantity
- Close other applications
- Consider using smaller model (llama3.2)

**❌ Import errors with LangChain**
- Solution: Update imports if needed
- Check LangChain version compatibility
- Try `pip install --upgrade langchain`

### Performance Tips:
1. **Clean Documents**: Well-formatted PDFs work best
2. **Specific Questions**: Clear, focused queries get better results
3. **Model Choice**: llama3 offers good balance of speed/quality
4. **Chunk Size**: 800 characters works well for most documents

---

## 🎯 Testing Your System

### Quick Test Scenarios:
1. **Basic Function**: "What are the penalty fares?"
2. **Specific Info**: "How much is a Zone 1+2 daily ticket?"
3. **Complex Query**: "What accessibility features are available on trams?"
4. **Edge Case**: Ask about something not in your documents

### Evaluation Process:
```bash
# Run comprehensive evaluation
python evaluate.py

# Results include:
# - Success rate across question categories
# - Average response times
# - Source retrieval effectiveness
# - Performance by difficulty level
```

---

## 📈 Project Extensions

### For Advanced Development:
- [ ] Add more evaluation metrics (ROUGE, BLEU scores)
- [ ] Implement conversation memory
- [ ] Create API endpoints
- [ ] Add multi-language support
- [ ] Build feedback collection system

### For Business Demo:
- [ ] Calculate time savings vs manual lookup
- [ ] Measure accuracy improvements
- [ ] Develop ROI analysis
- [ ] Create stakeholder presentations

---

## 🤝 Team Collaboration

### Git Workflow:
```bash
# Track changes
git add .
git commit -m "Add new evaluation metrics"
git push origin feature-branch

# Share improvements
# Document results in evaluation_results_*.csv files
# Update README with new findings
```

### Project Structure for Teams:
- **Backend Development**: Focus on `run_rag.py`, `utils/`
- **Frontend Development**: Enhance `app.py` interface
- **Evaluation**: Expand `evaluate.py`, add metrics
- **Documentation**: Update README, create presentation materials

---

## 🆘 Getting Help

### Quick Checks:
1. Is Ollama server running? (`ollama serve`)
2. Is virtual environment activated? (see `(rag-env)` in prompt)
3. Are documents in `data/` folder?
4. Are all Python files created correctly?

### Resources:
- **Ollama Documentation**: https://ollama.com/
- **LangChain Docs**: https://python.langchain.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Original Walert Project**: https://github.com/rmit-ir/walert

### Contact:
- Check console output for specific error messages
- Review this README for missed steps
- Test with sample documents first before using your own

---

## 📄 License & Usage

This project is for educational purposes (RMIT WIL). Ensure compliance with document licensing when using official transport documents.

**Built by**: [Your Team Name]  
**Date**: September 2025  
**Course**: Case Study in Data Science, RMIT University

---

## 🎊 Success Indicators

Your system is working correctly when:
- ✅ Web interface loads without errors
- ✅ Sample questions receive relevant answers
- ✅ Source documents are properly cited
- ✅ Evaluation shows >70% answer coverage
- ✅ Response times are under 10 seconds
- ✅ No "connection failed" errors

**Ready to demonstrate RAG value to stakeholders!** 🚀