import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.llms import VLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import base64
import re
import os
import tempfile

# Custom CSS styling for the application (unchanged)
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    .think-output {
        color: #ffcc00; /* Yellow color for <think> part */
        font-style: italic; /* Italic style for emphasis */
    }
    .actual-output {
        color: #00ff00; /* Green color for actual output */
        font-weight: bold; /* Bold style for emphasis */
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    .user-image {
        max-width: 300px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Application title and caption
st.title("🧠 AI Code Companion with RAG & vLLM")
st.caption("🚀 Your Multimodal AI Pair Programmer with Vision, RAG, & vLLM Superpowers")

# Sidebar configuration for model selection and settings
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["gemma3:4b", "deepscaler:latest"],  # Add models here
        index=0
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1
    )
    # New: File uploader for RAG documents
    uploaded_docs = st.file_uploader("Upload Documents for RAG (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True, key="doc_uploader")
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    - 👁️ Image Analysis
    - 📚 RAG Document Retrieval
    - ⚡ vLLM Acceleration
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/) | [vLLM](https://vllm.ai/)")

# Initialize session state for vector store
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# Initialize embedding model for RAG
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# Process uploaded documents for RAG
def process_documents(uploaded_docs):
    """Process uploaded documents and create a FAISS vector store."""
    if not uploaded_docs:
        return None
    documents = []
    for uploaded_doc in uploaded_docs:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_doc.name)[1]) as tmp_file:
            tmp_file.write(uploaded_doc.read())
            tmp_file_path = tmp_file.name
        # Load document based on file type
        if uploaded_doc.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_file_path)
        else:  # txt
            loader = TextLoader(tmp_file_path)
        documents.extend(loader.load())
        os.unlink(tmp_file_path)  # Clean up temp file
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Update vector store when new documents are uploaded
if uploaded_docs and not st.session_state.documents_processed:
    with st.spinner("📚 Processing documents..."):
        st.session_state.vectorstore = process_documents(uploaded_docs)
        st.session_state.documents_processed = True
        st.success("Documents processed successfully!")

# Modified system prompt to include RAG context
system_prompt = """
You are an expert AI coding assistant with vision and RAG capabilities. Provide concise, correct solutions with strategic print statements for debugging. Analyze images when provided. If relevant context is retrieved from documents, incorporate it into your response to enhance accuracy. Always respond in English.

Retrieved Context (if any): {context}
"""

# Manage session state for chat history
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm your AI Code Assistant with RAG & vLLM. Upload documents, ask coding questions, or analyze images! 💻📚👁️⚡"}]

# Create a container for the chat interface
chat_container = st.container()

# Function to stream and format AI output incrementally
def stream_formatted_output(raw_stream, placeholder):
    """Stream AI response token by token and format <think> and actual output."""
    full_response = ""
    for chunk in raw_stream:
        if chunk:
            full_response += chunk
            think_part = ""
            actual_part = ""
            if "<think>" in full_response and "</think>" in full_response:
                think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
                if think_match:
                    think_part = think_match.group(1)
                    actual_part = full_response.split("</think>")[1].strip()
            elif "<think>" in full_response:
                think_part = full_response.split("<think>")[1]
            else:
                actual_part = full_response
            html_output = ""
            if think_part:
                html_output += f'<div class="think-container"><span class="think-output"><think>{think_part}</think></span></div>'
            if actual_part:
                html_output += f'<div class="actual-container"><span class="actual-output">{actual_part}</span></div>'
            placeholder.markdown(html_output, unsafe_allow_html=True)
    return full_response

# Display chat messages in the container
with chat_container:
    for msg in st.session_state.message_log:
        with st.chat_message(msg["role"]):
            if msg["role"] == "ai":
                st.markdown(msg["content"], unsafe_allow_html=True)
            else:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item["type"] == "text":
                            st.markdown(item["text"])
                        elif item["type"] == "image_url":
                            st.markdown(f'<img src="{item["image_url"]}" class="user-image" alt="Uploaded Image">', unsafe_allow_html=True)
                else:
                    st.markdown(msg["content"])

# User input section: text query and optional image upload
user_query = st.chat_input("Type your coding question here...")
uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "png", "jpeg"], key="image_uploader")

# Function to build the prompt chain from message history
def build_prompt_chain(context=""):
    """Construct the prompt sequence for the AI model with RAG context."""
    prompt_sequence = [SystemMessage(content=system_prompt.format(context=context))]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessage(content=msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# Process user input and generate streamed AI response
if user_query:
    with st.spinner("🧠 Processing..."):
        # Handle multimodal input (text + optional image)
        image_content = None
        if uploaded_file is not None:
            file_type = uploaded_file.type
            image_data = uploaded_file.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            image_content = {"type": "image_url", "image_url": f"data:{file_type};base64,{image_base64}"}
            content = [
                {"type": "text", "text": user_query},
                image_content
            ]
        else:
            content = user_query
        
        # Append user message to the chat log
        st.session_state.message_log.append({"role": "user", "content": content})
        
        # Determine if uploads are present
        has_uploads = bool(uploaded_docs) or bool(uploaded_file)
        
        if has_uploads:
            # Use vLLM with gemma3:4b (assuming HF model path; adjust if needed, e.g., "google/gemma-2-2b-it")
            vllm_llm = VLLM(
                model="google/gemma-2-2b-it",  # Replace with actual HF path for 'gemma3:4b' if different
                temperature=temperature,
                max_tokens=512,
                dtype="half",  # For memory efficiency
                gpu_memory_utilization=0.95,
                trust_remote_code=True,
                stream=True
            )
            
            # Process documents sequentially with vLLM to generate summaries as context
            context = ""
            if uploaded_docs:
                for uploaded_doc in uploaded_docs:
                    # Load each document
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_doc.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_doc.read())
                        tmp_file_path = tmp_file.name
                    if uploaded_doc.name.endswith(".pdf"):
                        loader = PyPDFLoader(tmp_file_path)
                    else:
                        loader = TextLoader(tmp_file_path)
                    doc_pages = loader.load()
                    doc_content = "\n".join([page.page_content for page in doc_pages])
                    os.unlink(tmp_file_path)
                    
                    # Summarize with vLLM
                    summary_prompt = ChatPromptTemplate.from_template(
                        "Summarize this document for coding assistance context: {doc_content}"
                    )
                    summary_chain = summary_prompt | vllm_llm | StrOutputParser()
                    summary = ""
                    for chunk in summary_chain.stream({"doc_content": doc_content}):
                        summary += chunk
                    context += summary + "\n\n"
            
            # Use deepscaler:latest (Ollama) for final output, passing context + files (image) + user text
            deep_llm = ChatOllama(
                model="deepseek-r1:1.5b",
                base_url="http://localhost:11434",
                temperature=temperature,
                stream=True
            )
            
            # Retrieve additional RAG context if vectorstore exists
            rag_context = ""
            if st.session_state.vectorstore is not None:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.get_relevant_documents(user_query)
                rag_context = "\n".join([doc.page_content for doc in retrieved_docs])
            full_context = context + rag_context
            
            # Build prompt for deep_llm
            prompt_chain = build_prompt_chain(full_context)
            
            # For image, include in the final content for deep_llm (assuming deepscaler supports multimodal; if not, it will ignore)
            final_content = [{"type": "text", "text": user_query}]
            if image_content:
                final_content.append(image_content)
            
            processing_pipeline = prompt_chain | deep_llm | StrOutputParser()
            
            with st.chat_message("ai"):
                response_placeholder = st.empty()
                raw_stream = processing_pipeline.stream({})
                ai_response = stream_formatted_output(raw_stream, response_placeholder)
        
        else:
            # No uploads: Use original Ollama with selected model
            llm_engine = ChatOllama(
                model=selected_model,
                base_url="http://localhost:11434",
                temperature=temperature,
                stream=True
            )
            
            # Retrieve RAG context if available (even without new uploads, if previously processed)
            context = ""
            if st.session_state.vectorstore is not None:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.get_relevant_documents(user_query)
                context = "\n".join([doc.page_content for doc in retrieved_docs])
            
            # Generate and stream AI response
            prompt_chain = build_prompt_chain(context)
            processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
            
            with st.chat_message("ai"):
                response_placeholder = st.empty()
                raw_stream = processing_pipeline.stream({})
                ai_response = stream_formatted_output(raw_stream, response_placeholder)
        
        # Append the complete AI response to the chat log
        st.session_state.message_log.append({"role": "ai", "content": ai_response})
        
        # Rerun the app to update the chat display
        st.rerun()

