import streamlit as st
import requests
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration - Load from .env file
HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = os.getenv("API_URL", "https://router.huggingface.co/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2-2b-it")  # Model to use with router

# Page configuration
st.set_page_config(
    page_title="Stock Analysis AI Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 20%;
    }
    
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .message-content {
        line-height: 1.6;
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = ""

# System prompt for stock analysis
SYSTEM_PROMPT = """You are a friendly and knowledgeable stock market analysis assistant designed for beginners. 
Your role is to help users understand stock market concepts, analyze stocks, and make informed investment decisions.

Guidelines:
- Explain concepts in simple, easy-to-understand language
- Use examples and analogies when explaining complex topics
- Provide balanced analysis considering both risks and opportunities
- Always remind users that this is educational information, not financial advice
- Be encouraging and supportive to beginners
- Break down complex topics into digestible parts
- Use bullet points for clarity when listing multiple points

Topics you can help with:
- Stock fundamentals (P/E ratio, EPS, Market Cap, etc.)
- Technical analysis basics
- Investment strategies
- Market trends and sectors
- Risk management
- Portfolio diversification
- Reading financial statements
- Understanding market news

Always be helpful, clear, and educational in your responses."""

def query_huggingface(prompt, context=""):
    """Query the Hugging Face API using OpenAI-compatible chat completions format"""
    
    if not HF_API_KEY:
        return "❌ HF_API_KEY not found in .env file. Please add your Hugging Face API token."
    
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build messages array in OpenAI format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    # Add conversation history
    if context:
        history_lines = context.split('\n')
        for line in history_lines:
            if line.startswith("User:"):
                messages.append({"role": "user", "content": line.replace("User:", "").strip()})
            elif line.startswith("Assistant:"):
                messages.append({"role": "assistant", "content": line.replace("Assistant:", "").strip()})
    
    # Add current user message
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        # Handle different status codes
        if response.status_code == 503:
            return "⏳ Model is currently loading. Please try again in a moment."
        
        if response.status_code == 401:
            return "❌ Invalid API key. Please check your HF_API_KEY in the .env file."
        
        if response.status_code == 429:
            return "⚠️ Rate limit reached. Please wait a moment and try again."
        
        response.raise_for_status()
        result = response.json()
        
        # Extract response from OpenAI format
        if "choices" in result and len(result["choices"]) > 0:
            message_content = result["choices"][0].get("message", {}).get("content", "")
            if message_content:
                return message_content.strip()
            else:
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        else:
            return f"⚠️ Unexpected response format: {result}"
    
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if "401" in error_msg:
            return "❌ Authentication failed. Please verify your HF_API_KEY."
        elif "404" in error_msg:
            return f"❌ Model '{MODEL_NAME}' not found. Please check your MODEL_NAME in .env file."
        else:
            return f"⚠️ Connection error: {error_msg}\n\nPlease check:\n- Your internet connection\n- Your HF_API_KEY\n- The API endpoint URL"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-info"><h2>📈 Stock Analysis AI</h2><p>Your beginner-friendly investment assistant</p></div>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Quick Start Guide")
    st.markdown("""
    <div class="feature-box">
    <strong>How to use:</strong><br>
    1. Type your stock market question<br>
    2. Click Send or press Enter<br>
    3. Get beginner-friendly explanations<br>
    4. Ask follow-up questions anytime
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What is a P/E ratio?",
        "How do I start investing in stocks?",
        "What's the difference between stocks and bonds?",
        "Explain market capitalization",
        "What are blue-chip stocks?",
        "How to read a stock chart?",
        "What is dollar-cost averaging?",
        "Explain diversification"
    ]
    
    for question in example_questions:
        if st.button(question, key=question):
            st.session_state.current_question = question
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_context = ""
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div class="feature-box">
    <strong>⚠️ Disclaimer:</strong><br>
    This chatbot provides educational information only. 
    Always consult with a licensed financial advisor before making investment decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # API Status
    st.markdown("### 🔌 API Status")
    if HF_API_KEY:
        st.success("✅ API Key Loaded")
    else:
        st.error("❌ API Key Missing")
    
    st.info(f"🔗 Endpoint: {API_URL.split('/')[-2] + '/' + API_URL.split('/')[-1]}")
    st.info(f"🤖 Model: {MODEL_NAME}")
    
    if not HF_API_KEY:
        st.warning("⚠️ Update your .env file with:")
        st.code("""HF_API_KEY=your_token_here
API_URL=https://router.huggingface.co/v1/chat/completions
MODEL_NAME=google/gemma-2-2b-it""", language="text")

# Main content
st.markdown('<h1 class="main-header">📊 Stock Analysis AI Assistant</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; margin-bottom: 2rem;'>Ask me anything about stock market analysis - I'm here to help beginners learn!</p>", unsafe_allow_html=True)

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        icon = "🧑" if message["role"] == "user" else "🤖"
        
        st.markdown(f"""
        <div class="chat-message {role_class}">
            <div class="message-header">{icon} {message["role"].capitalize()}</div>
            <div class="message-content">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

# Handle example question click
if "current_question" in st.session_state:
    user_input = st.session_state.current_question
    del st.session_state.current_question
else:
    user_input = None

# Chat input
col1, col2 = st.columns([6, 1])

with col1:
    prompt = st.text_input(
        "Your Question:",
        placeholder="e.g., What is the difference between value and growth stocks?",
        key="user_input_field",
        value=user_input if user_input else "",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send 📤")

# Process user input
if (send_button or prompt) and prompt and HF_API_KEY:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Build conversation context (last 3 exchanges)
    context_messages = st.session_state.messages[-6:] if len(st.session_state.messages) > 6 else st.session_state.messages
    context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in context_messages[:-1]])
    
    # Get AI response
    with st.spinner("🤔 Analyzing your question..."):
        response = query_huggingface(prompt, context)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Update context
    st.session_state.conversation_context = context
    
    # Rerun to display new messages
    st.rerun()

elif (send_button or prompt) and not HF_API_KEY:
    st.error("⚠️ Please add your Hugging Face API key to the .env file")

# Welcome message if no conversation
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 1rem; margin: 2rem 0;'>
        <h2 style='color: #667eea;'>👋 Welcome to Your Stock Analysis Assistant!</h2>
        <p style='color: #666; font-size: 1.1rem; margin-top: 1rem;'>
            I'm here to help you understand stock market concepts and analysis.<br>
            Click on any example question in the sidebar or type your own question below!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box" style='text-align: center;'>
            <h3 style='color: #667eea;'>📚 Beginner Friendly</h3>
            <p>Simple explanations for complex concepts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box" style='text-align: center;'>
            <h3 style='color: #667eea;'>💬 Interactive Chat</h3>
            <p>Ask follow-up questions anytime</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box" style='text-align: center;'>
            <h3 style='color: #667eea;'>🎯 Comprehensive</h3>
            <p>Cover all aspects of stock analysis</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #999; font-size: 0.9rem;'>
    Powered by Google Gemma-2-2b-it via Hugging Face | Built with Streamlit<br>
    <em>Remember: This is for educational purposes only. Always do your own research!</em>
</p>
""", unsafe_allow_html=True)