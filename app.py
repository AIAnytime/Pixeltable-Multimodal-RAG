"""
Pixeltable Multimodal Search Application
A practical demonstration of Pixeltable's unified approach to multimodal AI
"""

import streamlit as st
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.functions import openai, huggingface
from pixeltable.iterators import DocumentSplitter
import os
from dotenv import load_dotenv
from pathlib import Path
import tempfile
from PIL import Image
import io
import graphviz

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Pixeltable Multimodal Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Pixeltable directory
PIXELTABLE_DIR = Path.home() / ".pixeltable_demo"
os.environ['PIXELTABLE_HOME'] = str(PIXELTABLE_DIR)


@st.cache_resource
def init_pixeltable():
    """Initialize Pixeltable environment - cached to run only once"""
    try:
        # Check if directory exists, create if it doesn't
        existing_dirs = pxt.list_dirs()
        if 'demo' not in existing_dirs:
            pxt.create_dir('demo')
        return True
    except Exception as e:
        # Directory might already exist, which is fine
        if "already exists" in str(e).lower():
            return True
        # Don't show error for harmless issues
        return True


def create_architecture_diagram():
    """Create Pixeltable architecture diagram using Graphviz"""
    dot = graphviz.Digraph(comment='Pixeltable Architecture')
    dot.attr(rankdir='LR', bgcolor='transparent', size='10,6')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Application layer
    dot.node('app', 'Your\nApplication', fillcolor='#667eea', fontcolor='white', width='1.5', height='0.8')
    
    # Traditional stack (left path)
    with dot.subgraph(name='cluster_traditional') as c:
        c.attr(label='Traditional Stack (Complex)', style='dashed', color='#e74c3c', fontcolor='#e74c3c', fontsize='11')
        c.node('vec_db', 'Vector DB\n(Pinecone)', fillcolor='#ffcccc', width='1.5')
        c.node('sql_db', 'SQL DB\n(PostgreSQL)', fillcolor='#ffcccc', width='1.5')
        c.node('obj_store', 'Object Storage\n(S3)', fillcolor='#ffcccc', width='1.5')
        c.node('pipeline', 'Pipeline\n(Airflow)', fillcolor='#ffcccc', width='1.5')
        c.node('glue', 'Glue Code\n(Scripts)', fillcolor='#ff9999', width='1.5')
    
    # Pixeltable stack (right path)
    with dot.subgraph(name='cluster_pixeltable') as c:
        c.attr(label='Pixeltable Stack (Simple)', style='filled', fillcolor='#e8f5e9', color='#27ae60', fontcolor='#27ae60', fontsize='11')
        c.node('pxt', 'Pixeltable\nUnified Interface', fillcolor='#81c784', fontcolor='white', width='2', height='1')
        c.node('features', 'Built-in:\n• Auto-Computation\n• Version Control\n• Vector Search\n• ML Integration', 
               fillcolor='#a5d6a7', shape='note', width='2')
    
    # Edges for traditional (showing complexity)
    dot.edge('app', 'vec_db', label='embeddings', color='#e74c3c', style='dashed')
    dot.edge('app', 'sql_db', label='metadata', color='#e74c3c', style='dashed')
    dot.edge('app', 'obj_store', label='files', color='#e74c3c', style='dashed')
    dot.edge('app', 'pipeline', label='orchestrate', color='#e74c3c', style='dashed')
    dot.edge('app', 'glue', label='integrate', color='#e74c3c', style='dashed')
    
    # Edges for Pixeltable (showing simplicity)
    dot.edge('app', 'pxt', label='single API', color='#27ae60', penwidth='3')
    dot.edge('pxt', 'features', style='dotted', color='#27ae60')
    
    return dot


def create_simple_flow_diagram():
    """Create a simple data flow diagram for Pixeltable"""
    dot = graphviz.Digraph(comment='Pixeltable Data Flow')
    dot.attr(rankdir='TB', bgcolor='transparent')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9', penwidth='2')
    
    # Data sources
    dot.node('docs', 'Documents\n(PDF, TXT)', fillcolor='#e3f2fd', shape='folder')
    dot.node('images', 'Images\n(JPG, PNG)', fillcolor='#e3f2fd', shape='folder')
    dot.node('videos', 'Videos\n(MP4)', fillcolor='#e3f2fd', shape='folder')
    
    # Pixeltable
    dot.node('pxt', 'Pixeltable\nTables', fillcolor='#81c784', fontcolor='white', width='2', height='1')
    
    # Processing
    dot.node('compute', 'Computed Columns\n(Auto-Processing)', fillcolor='#fff9c4', width='2.5')
    
    # Outputs
    dot.node('embed', 'Embeddings', fillcolor='#f3e5f5', shape='cylinder')
    dot.node('analysis', 'AI Analysis', fillcolor='#f3e5f5', shape='cylinder')
    dot.node('search', 'Vector Search', fillcolor='#f3e5f5', shape='cylinder')
    
    # Flow
    dot.edge('docs', 'pxt', color='#1976d2')
    dot.edge('images', 'pxt', color='#1976d2')
    dot.edge('videos', 'pxt', color='#1976d2')
    dot.edge('pxt', 'compute', label='automatic', color='#27ae60')
    dot.edge('compute', 'embed', color='#7b1fa2')
    dot.edge('compute', 'analysis', color='#7b1fa2')
    dot.edge('compute', 'search', color='#7b1fa2')
    
    return dot


@st.cache_resource
def create_document_rag_system():
    """Create a complete RAG system for documents - cached to run only once"""
    try:
        # Try to get existing tables, create if they don't exist
        try:
            docs = pxt.get_table('demo.docs')
        except:
            docs = pxt.create_table('demo.docs', {'doc': pxt.Document})
        
        # Create chunks view
        try:
            chunks = pxt.get_table('demo.doc_chunks')
        except:
            chunks = pxt.create_view(
                'demo.doc_chunks',
                docs,
                iterator=DocumentSplitter.create(
                    document=docs.doc,
                    separators='sentence'
                )
            )
            # Add embedding index immediately after creating view
            embed_model = huggingface.sentence_transformer.using(model_id='sentence-transformers/all-MiniLM-L6-v2')
            chunks.add_embedding_index('text', string_embed=embed_model)
        
        # Create QA table
        try:
            qa = pxt.get_table('demo.qa_system')
        except:
            qa = pxt.create_table('demo.qa_system', {'prompt': pxt.String})
            
            # Add computed columns immediately after creating table
            # 1. Add context retrieval
            @pxt.query
            def get_relevant_context(query_text: str, limit: int = 3):
                sim = chunks.text.similarity(query_text)
                return chunks.order_by(sim, asc=False).limit(limit).select(chunks.text)
            
            qa.add_computed_column(context=get_relevant_context(qa.prompt, 3))
            
            # 2. Format final prompt
            qa.add_computed_column(
                final_prompt=pxtf.string.format(
                    """
                    PASSAGES:
                    {0}

                    QUESTION:
                    {1}
                    
                    Provide a detailed answer based on the passages above.
                    """,
                    qa.context,
                    qa.prompt
                )
            )
            
            # 3. Add OpenAI completion
            qa.add_computed_column(
                answer=openai.chat_completions(
                    model='gpt-4o-mini',
                    messages=[{'role': 'user', 'content': qa.final_prompt}]
                ).choices[0].message.content
            )
        
        return docs, chunks, qa
    except Exception as e:
        st.error(f"Error creating RAG system: {e}")
        import traceback
        with st.expander("Show detailed error"):
            st.code(traceback.format_exc())
        # Return None to indicate failure
        return None, None, None


@st.cache_resource
def create_image_analysis_system():
    """Create image analysis system - cached to run only once"""
    try:
        # Try to get existing table, create if it doesn't exist
        try:
            images = pxt.get_table('demo.images')
        except:
            images = pxt.create_table('demo.images', {'input_image': pxt.Image})
            
            # Add vision analysis
            images.add_computed_column(
                vision_description=openai.vision(
                    model='gpt-4o-mini',
                    prompt="Describe this image in detail, including objects, colors, and context.",
                    image=images.input_image
                )
            )
            
            # Add embedding index for image similarity search
            clip_model = huggingface.clip_image.using(model_id='openai/clip-vit-base-patch32')
            images.add_embedding_index('input_image', image_embed=clip_model)
        
        return images
    except Exception as e:
        st.error(f"Error creating image system: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Navigation Menu",
        ["Home", "Document RAG", "Image Analysis", "Why Pixeltable?"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### API Status")
    if os.getenv("OPENAI_API_KEY"):
        st.success("OpenAI API Key Loaded")
    else:
        st.error("OpenAI API Key Missing")
    
    st.markdown("---")
    st.markdown("""
    ### Resources
    - [Pixeltable Docs](https://pixeltable.readme.io/)
    - [GitHub](https://github.com/pixeltable/pixeltable)
    """)


# Main content
if page == "Home":
    st.markdown('<h1 class="main-header">Pixeltable Multimodal Demo</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## What is Pixeltable?
    
    **Pixeltable** is a revolutionary framework that unifies multimodal AI development into a single, 
    declarative table-based interface. It eliminates the complexity of managing separate:
    - Vector databases
    - SQL databases
    - Object storage
    - Pipeline orchestrators
    - API wrappers
    
    ### Key Benefits
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>Unified Interface</h3>
            <p>Everything in tables: embeddings, images, videos, LLM outputs, metadata</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>Auto-Computation</h3>
            <p>Calculated columns update automatically when data changes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>Version Control</h3>
            <p>Built-in versioning and time-travel capabilities</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Architecture diagram
    st.markdown("## Architecture")
    
    tab1, tab2 = st.tabs(["Stack Comparison", "Data Flow"])
    
    with tab1:
        st.markdown("**Traditional Stack vs Pixeltable**")
        diagram = create_architecture_diagram()
        st.graphviz_chart(diagram)
    
    with tab2:
        st.markdown("**How Data Flows Through Pixeltable**")
        flow_diagram = create_simple_flow_diagram()
        st.graphviz_chart(flow_diagram)
    
    st.markdown("""
    ## Use Cases
    
    1. **RAG Systems** - Document Q&A without external vector databases
    2. **Image Search** - Multimodal similarity search
    3. **Video Analysis** - Frame extraction and processing
    4. **Agent Workflows** - Tool calls and conversation history
    5. **Dataset Preparation** - ML dataset creation and versioning
    
    ### Select a demo from the sidebar to get started!
    """)

elif page == "Document RAG":
    st.markdown("# Document RAG System")
    
    st.info("""
    **How it works:**
    1. Upload a PDF or text document
    2. Pixeltable automatically chunks and embeds it
    3. Ask questions and get AI-powered answers based on your document
    """)
    
    # Initialize system
    docs, chunks, qa = None, None, None
    
    # Always try to initialize
    init_pixeltable()
    result = create_document_rag_system()
    if result and result != (None, None, None):
        docs, chunks, qa = result
    
    if docs is not None and chunks is not None and qa is not None:
        tab1, tab2 = st.tabs(["Upload Document", "Ask Questions"])
        
        with tab1:
            st.markdown("### Upload Document")
            
            # Option to use sample document
            use_sample = st.checkbox("Use sample document (Jefferson Amazon PDF)")
            
            if use_sample:
                sample_url = 'https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/Jefferson-Amazon.pdf'
                if st.button("Load Sample Document"):
                    with st.spinner("Loading sample document..."):
                        try:
                            docs.insert([{'doc': sample_url}])
                            st.success("Sample document loaded successfully!")
                            
                            # Update session state counts
                            chunk_count = len(chunks.select().collect())
                            st.session_state.doc_count = len(docs.select().collect())
                            st.session_state.chunk_count = chunk_count
                            
                            if chunk_count == 0:
                                st.warning(f"⚠️ Created {chunk_count} text chunks")
                            else:
                                st.info(f"✅ Created {chunk_count} text chunks - Ready for questions!")
                        except Exception as e:
                            st.error(f"Error loading sample: {e}")
            
            uploaded_file = st.file_uploader(
                "Or upload your own PDF or text file",
                type=['pdf', 'txt'],
                help="Upload a document to analyze"
            )
            
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        try:
                            docs.insert([{'doc': tmp_path}])
                            st.success(f"Document '{uploaded_file.name}' processed successfully!")
                            
                            # Update session state counts
                            chunk_count = len(chunks.select().collect())
                            st.session_state.doc_count = len(docs.select().collect())
                            st.session_state.chunk_count = chunk_count
                            
                            if chunk_count == 0:
                                st.warning(f"⚠️ Created {chunk_count} text chunks - The document may be empty, contain only images, or be a scanned PDF. Try uploading a text-based document.")
                            else:
                                st.info(f"✅ Created {chunk_count} text chunks - Ready for questions!")
                        except Exception as e:
                            st.error(f"Error processing document: {e}")
        
        with tab2:
            st.markdown("### Ask Questions")
            
            # Use session state to cache counts and avoid async conflicts
            if 'doc_count' not in st.session_state:
                st.session_state.doc_count = 0
            
            if 'chunk_count' not in st.session_state:
                st.session_state.chunk_count = 0
            
            # Add refresh button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"{st.session_state.doc_count} document(s) with {st.session_state.chunk_count} text chunks available")
            with col2:
                if st.button("🔄 Refresh", help="Refresh document and chunk counts"):
                    try:
                        st.session_state.doc_count = len(docs.select().collect())
                        st.session_state.chunk_count = len(chunks.select().collect())
                        st.rerun()
                    except Exception as e:
                        st.error(f"Refresh failed: {e}")
            
            doc_count = st.session_state.doc_count
            chunk_count = st.session_state.chunk_count
            
            if doc_count > 0 and chunk_count > 0:
                # Add suggested questions for sample document
                st.markdown("**Try these sample questions:**")
                sample_questions = [
                    "What can you tell me about Amazon?",
                    "Who is Thomas Jefferson?",
                    "What is the main topic of this document?"
                ]
                
                question = st.text_input(
                    "Enter your question:",
                    placeholder="What is this document about?"
                )
                
                # Quick buttons for sample questions
                cols = st.columns(3)
                for idx, sq in enumerate(sample_questions):
                    if cols[idx].button(sq, key=f"sq_{idx}"):
                        question = sq
                
                if st.button("Get Answer") and question:
                    with st.spinner("Searching and generating answer..."):
                        try:
                            qa.insert([{'prompt': question}])
                            result = qa.select(qa.answer).tail(1)
                            
                            if result and len(result) > 0:
                                st.markdown("### Answer")
                                st.markdown(result[0]['answer'])
                        except Exception as e:
                            st.error(f"Error: {e}")
                            import traceback
                            st.error(traceback.format_exc())
            elif doc_count == 0:
                st.warning("Please upload a document first!")
            else:
                st.warning("Your document has no text chunks. Please upload a text-based document (not a scanned image PDF).")
    else:
        st.warning("Document RAG system is initializing... If you see this message after uploading documents, check the error details above.")

elif page == "Image Analysis":
    st.markdown("# Image Analysis")
    
    st.info("""
    **How it works:**
    1. Upload an image (JPG, PNG)
    2. GPT-4 Vision automatically analyzes it
    3. Results are stored in Pixeltable for future search
    """)
    
    # Initialize system
    images = None
    with st.spinner("Initializing Pixeltable..."):
        if init_pixeltable():
            images = create_image_analysis_system()
    
    if images:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Upload Image")
            
            # Option to use sample images
            use_sample = st.checkbox("Use sample images")
            
            if use_sample:
                # Using reliable image URLs
                sample_images = [
                    ('Cat', 'https://raw.github.com/pixeltable/pixeltable/release/docs/resources/images/000000000025.jpg'),
                    ('Sample 2', 'https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=800&q=80'),
                    ('Sample 3', 'https://images.unsplash.com/photo-1425082661705-1834bfd09dca?w=800&q=80')
                ]
                
                selected_sample = st.selectbox("Select a sample image:", [name for name, _ in sample_images])
                
                if st.button("Load Sample Image"):
                    sample_url = next(url for name, url in sample_images if name == selected_sample)
                    with st.spinner("Loading and analyzing sample image..."):
                        try:
                            images.insert([{'input_image': sample_url}])
                            result = images.select(images.vision_description).tail(1)
                            
                            if result and len(result) > 0:
                                st.success(f"Sample image '{selected_sample}' analyzed!")
                                st.markdown("### Analysis Result")
                                st.write(result[0]['vision_description'])
                        except Exception as e:
                            st.error(f"Error loading sample image: {e}")
                            st.info("💡 Tip: If sample images fail to load, please upload your own image file instead using the file uploader below.")
            
            uploaded_image = st.file_uploader(
                "Or upload your own image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image to analyze"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", width='stretch')
                
                if st.button("Analyze Image"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image.save(tmp_file, format='JPEG')
                        tmp_path = tmp_file.name
                    
                    with st.spinner("Analyzing image with GPT-4 Vision..."):
                        try:
                            images.insert([{'input_image': tmp_path}])
                            result = images.select(images.vision_description).tail(1)
                            
                            if result and len(result) > 0:
                                with col2:
                                    st.markdown("### Analysis Result")
                                    st.success(result[0]['vision_description'])
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        # Show gallery
        st.markdown("---")
        st.markdown("### Image Gallery")
        try:
            # Get the last 6 images - tail() returns results directly
            all_images = images.select(images.input_image, images.vision_description).tail(6)
            
            if all_images and len(all_images) > 0:
                cols = st.columns(3)
                for idx, img_data in enumerate(all_images):
                    with cols[idx % 3]:
                        st.image(img_data['input_image'], width='stretch')
                        with st.expander("View Analysis"):
                            st.write(img_data['vision_description'])
            else:
                st.info("No images analyzed yet. Upload one to get started!")
        except Exception as e:
            st.error(f"Error loading gallery: {e}")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.error("Failed to initialize Image Analysis system. Please refresh the page.")

elif page == "Why Pixeltable?":
    st.markdown("# Why Pixeltable?")
    
    st.markdown("## Architecture Comparison")
    st.markdown("**Traditional approach requires multiple services. Pixeltable unifies everything.**")
    
    # Show architecture comparison diagram
    arch_diagram = create_architecture_diagram()
    st.graphviz_chart(arch_diagram)
    
    st.markdown("---")
    
    st.markdown("""
    ## The Problem with Traditional Approaches
    
    Building multimodal AI applications typically requires piecing together:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Traditional Stack
        - **Pinecone/Weaviate** for embeddings
        - **PostgreSQL** for metadata
        - **S3/MinIO** for large files
        - **Airflow** for pipelines
        - **Python scripts** everywhere
        - **Multiple API wrappers** (OpenAI, HuggingFace, etc.)
        
        **Result:** Complex, fragile, hard to maintain
        """)
    
    with col2:
        st.markdown("""
        ### Pixeltable Approach
        - **Single unified table** for everything
        - **Automatic computation** on data changes
        - **Built-in versioning** and rollback
        - **Declarative interface** - describe what, not how
        - **Integrated ML models** out of the box
        - **No glue code** needed
        
        **Result:** Simple, maintainable, powerful
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ## Real-World Impact
    
    ### 1. Development Speed
    - **30 lines of code** for a complete RAG system
    - **No infrastructure setup** required
    - **Instant prototyping** and iteration
    
    ### 2. Maintenance
    - **Single source of truth** for all data
    - **Automatic updates** when schema changes
    - **Easy debugging** with built-in lineage
    
    ### 3. Cost Efficiency
    - **Smart caching** - only recompute what changed
    - **No redundant API calls**
    - **Efficient storage** with automatic optimization
    
    ### 4. Scalability
    - **Production-ready** from day one
    - **Easy to add** new data sources
    - **Version control** for reproducibility
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## Comparison Table
    """)
    
    import pandas as pd
    
    comparison_data = {
        'Feature': ['Setup Time', 'Lines of Code', 'External Dependencies', 'Vector DB', 'Version Control', 'Automatic Updates', 'Learning Curve'],
        'Traditional': ['Hours', '500+', 'Multiple', 'Required', 'Manual', 'No', 'Steep'],
        'Pixeltable': ['Minutes', '~30', 'One', 'Built-in', 'Built-in', 'Yes', 'Gentle']
    }
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    st.markdown("""
    ## Get Started
    
    ```bash
    # Install Pixeltable
    pip install pixeltable
    
    # Create your first table
    import pixeltable as pxt
    t = pxt.create_table('my_images', {'img': pxt.Image})
    
    # Add AI capabilities
    from pixeltable.functions import openai
    t.add_computed_column(
        description=openai.vision(
            model='gpt-4o-mini',
            prompt="Describe this image",
            image=t.img
        )
    )
    
    # Insert and automatically process
    t.insert([{'img': 'path/to/image.jpg'}])
    ```
    
    ### Learn More
    - [Official Documentation](https://pixeltable.readme.io/)
    - [GitHub Repository](https://github.com/pixeltable/pixeltable)
    - [Example Notebooks](https://github.com/pixeltable/pixeltable/tree/main/docs/tutorials)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Pixeltable | © 2024</p>
</div>
""", unsafe_allow_html=True)
