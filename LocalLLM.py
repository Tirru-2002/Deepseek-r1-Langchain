import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import base64
import re

# Custom CSS styling for the application
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
st.title("🧠 AI Code Companion")
st.caption("🚀 Your Multimodal AI Pair Programmer with Vision & Debugging Superpowers")

# Sidebar configuration for model selection and settings
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["gemma3:4b", "deepscaler:latest"], #Add models Here, which you have downloaded from ollama.
        index=0
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    - 👁️ Image Analysis
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initialize the chat engine with streaming enabled
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=temperature,
    stream=True  # Enable streaming for token-by-token response
)

# Define the system prompt
system_prompt = "You are an expert AI coding assistant with vision capabilities. Provide concise, correct solutions with strategic print statements for debugging. Analyze images when provided and incorporate them into your responses. Always respond in English."

# Manage session state for chat history
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm your Ai code Assistant. How can I help you code or analyze images today? 💻👁️"}]

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
            # Parse <think> and actual output sections
            if "<think>" in full_response and "</think>" in full_response:
                think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
                if think_match:
                    think_part = think_match.group(1)
                    actual_part = full_response.split("</think>")[1].strip()
            elif "<think>" in full_response:
                think_part = full_response.split("<think>")[1]
            else:
                actual_part = full_response
            # Build HTML output with styled sections
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
def build_prompt_chain():
    """Construct the prompt sequence for the AI model."""
    prompt_sequence = [SystemMessage(content=system_prompt)]
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
        if uploaded_file is not None:
            file_type = uploaded_file.type
            image_data = uploaded_file.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            content = [
                {"type": "text", "text": user_query},
                {"type": "image_url", "image_url": f"data:{file_type};base64,{image_base64}"}
            ]
        else:
            content = user_query
        
        # Append user message to the chat log
        st.session_state.message_log.append({"role": "user", "content": content})
        
        # Generate and stream AI response
        prompt_chain = build_prompt_chain()
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        
        with st.chat_message("ai"):
            response_placeholder = st.empty()  # Placeholder for streaming response
            raw_stream = processing_pipeline.stream({})
            ai_response = stream_formatted_output(raw_stream, response_placeholder)
        
        # Append the complete AI response to the chat log
        st.session_state.message_log.append({"role": "ai", "content": ai_response})
        
        # Rerun the app to update the chat display
        st.rerun()
