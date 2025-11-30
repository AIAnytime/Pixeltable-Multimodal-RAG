# 🎯 Pixeltable Multimodal Demo

A comprehensive demonstration of **Pixeltable** - the unified framework for multimodal AI applications.

![Pixeltable Architecture](https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/source/data/pixeltable-overview.png)

## 🚀 What is Pixeltable?

**Pixeltable** revolutionizes multimodal AI development by unifying everything into a single table-based interface:

- 📄 **Documents** with automatic chunking and embedding
- 🖼️ **Images** with vision AI analysis
- 🎥 **Videos** with frame extraction
- 🎵 **Audio** with transcription
- 🤖 **LLM Integration** (OpenAI, HuggingFace)
- 🔍 **Vector Search** built-in
- 📊 **RAG Systems** in ~30 lines of code

### ❌ Traditional Approach
- Multiple services (Pinecone, PostgreSQL, S3)
- Complex pipeline orchestration (Airflow)
- 500+ lines of glue code
- Difficult to maintain and debug

### ✅ Pixeltable Approach
- Single unified system
- Automatic computation
- ~30 lines of code
- Built-in version control

## 📦 Installation

### Prerequisites
- Python 3.9+
- OpenAI API key (in `.env` file)

### Quick Setup

```bash
# 1. Make setup script executable
chmod +x setup.sh

# 2. Run setup (creates .venv and installs dependencies)
./setup.sh

# 3. Activate virtual environment
source .venv/bin/activate
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Running the Demo

### 1. Web Application (Streamlit)

Beautiful interactive web interface with:
- 📄 Document RAG Q&A
- 🖼️ Image analysis with GPT-4 Vision
- 💡 Educational content about Pixeltable

```bash
streamlit run app.py
```

**Features:**
- Upload documents (PDF, TXT) and ask questions
- Upload images for AI-powered analysis
- Visual comparison of traditional vs Pixeltable approach
- Live examples of multimodal processing

### 2. Jupyter Notebook

Step-by-step tutorial with executable examples:

```bash
jupyter notebook pixeltable_demo.ipynb
```

**Contents:**
1. ✅ Setup and installation
2. 💰 Automatic profit calculation example
3. 🖼️ Image analysis with GPT-4 Vision
4. 🔍 Image similarity search with CLIP
5. 📄 Complete RAG system (30 lines!)
6. 🔄 Incremental updates demonstration
7. 🎨 Custom user-defined functions

## 📁 Project Structure

```
pixeltable-demo/
├── app.py                    # Streamlit web application
├── pixeltable_demo.ipynb     # Jupyter notebook tutorial
├── requirements.txt          # Python dependencies
├── setup.sh                  # Setup script
├── .env                      # API keys (not in git)
├── README.md                 # This file
└── project.md                # Project documentation
```

## 🎯 Key Features Demonstrated

### 1. Automatic Computation
```python
# Add computed column - automatically calculates for ALL rows
films.add_computed_column(profit=(films.revenue - films.budget))
```

### 2. Image Analysis
```python
# Add GPT-4 Vision - automatically analyzes all images
images.add_computed_column(
    vision_description=openai.vision(
        model='gpt-4o-mini',
        prompt="Describe this image",
        image=images.input_image
    )
)
```

### 3. Complete RAG System
```python
# Create documents table
docs = pxt.create_table('docs', {'doc': pxt.Document})

# Auto-chunk documents
chunks = pxt.create_view('chunks', docs, 
    iterator=DocumentSplitter.create(document=docs.doc, separators='sentence'))

# Add embedding index using .using()
embed_model = huggingface.sentence_transformer.using(model_id='all-MiniLM-L6-v2')
chunks.add_embedding_index('text', string_embed=embed_model)

# Create Q&A with automatic answer generation
qa = pxt.create_table('qa', {'prompt': pxt.String})
qa.add_computed_column(answer=openai.chat_completions(...))
```

**That's it! ~30 lines for a complete RAG system!**

## 💡 Use Cases

1. **📄 Document Q&A (RAG)**
   - Legal document analysis
   - Research paper exploration
   - Knowledge base search

2. **🖼️ Image Search**
   - Product catalog search
   - Medical image retrieval
   - Content moderation

3. **🎥 Video Analysis**
   - Content summarization
   - Scene detection
   - Automated tagging

4. **🤖 AI Agents**
   - Tool call history
   - Conversation tracking
   - State management

5. **📊 Dataset Preparation**
   - ML dataset creation
   - Data versioning
   - Feature engineering

## 🔑 Environment Setup

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## 🎓 Learning Path

### For Beginners
1. Start with the **Streamlit app** (`streamlit run app.py`)
2. Explore the "Why Pixeltable?" section
3. Try uploading a document and asking questions

### For Developers
1. Open the **Jupyter notebook** (`jupyter notebook pixeltable_demo.ipynb`)
2. Execute each cell to see how Pixeltable works
3. Modify examples with your own data
4. Build your own multimodal application!

## 🏗️ Architecture

### Traditional ML Stack
```
Application
    ↓
┌─────────────────────────────────────┐
│ Pinecone/Weaviate (Vector DB)      │
│ PostgreSQL (Metadata)               │
│ S3/MinIO (Object Storage)           │
│ Airflow (Pipeline Orchestration)    │
│ Python Scripts (Glue Code)          │
│ API Wrappers (OpenAI, HuggingFace)  │
└─────────────────────────────────────┘
```

### Pixeltable Stack
```
Application
    ↓
┌─────────────────────────────────────┐
│ Pixeltable (Everything!)            │
│  - Tables with computed columns     │
│  - Automatic updates                │
│  - Built-in versioning              │
└─────────────────────────────────────┘
```

## 📊 Performance Benefits

| Metric | Traditional | Pixeltable |
|--------|-------------|------------|
| Setup Time | Hours | Minutes |
| Lines of Code (RAG) | 500+ | ~30 |
| External Services | 5+ | 1 |
| Maintenance Complexity | High | Low |
| Version Control | Manual | Built-in |
| Incremental Updates | Custom Code | Automatic |

## 🚧 Common Issues & Solutions

### Issue: OpenAI API Key Error
```bash
# Solution: Ensure .env file exists with correct key
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Issue: Module Not Found
```bash
# Solution: Activate virtual environment
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: Port Already in Use (Streamlit)
```bash
# Solution: Use different port
streamlit run app.py --server.port 8502
```

## 📚 Resources

- **Official Docs**: https://pixeltable.readme.io/
- **GitHub**: https://github.com/pixeltable/pixeltable
- **Examples**: https://github.com/pixeltable/pixeltable/tree/main/docs/tutorials
- **Discord**: https://discord.gg/pixeltable

## 🎯 Next Steps

1. **Experiment**: Modify the notebook examples with your data
2. **Build**: Create your own multimodal application
3. **Share**: Share your Pixeltable projects!

## 🤝 Contributing

This is a demo project. For contributions to Pixeltable itself:
- Visit https://github.com/pixeltable/pixeltable
- Check their contribution guidelines

## 📝 License

This demo project is MIT licensed. Pixeltable itself has its own license.

## 🙏 Acknowledgments

- **Pixeltable Team** for creating this amazing framework
- **OpenAI** for GPT-4 and Vision APIs
- **HuggingFace** for open-source models

---

## 💻 Quick Commands Reference

```bash
# Setup
./setup.sh
source .venv/bin/activate

# Run Streamlit app
streamlit run app.py

# Run Jupyter notebook
jupyter notebook pixeltable_demo.ipynb

# Install new package
pip install package-name
pip freeze > requirements.txt

# Deactivate virtual environment
deactivate
```

---

**Built with ❤️ using Pixeltable**

For questions or issues, please refer to the [Pixeltable documentation](https://pixeltable.readme.io/).
