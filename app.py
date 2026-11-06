import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch

# 1. Page Configuration
st.set_page_config(
    page_title="AI Sentiment & Generation Hub", 
    page_icon="✨", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS with Gorgeous Gradients and Modern Styling
st.markdown("""
    <style>
    /* Main Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        color: #f8fafc;
    }
    
    /* Custom Gradient Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        transform: translateY(-2px);
    }

    /* Headers and Text Styling */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Header Section
st.title("✨ AI Sentiment & Generation Hub")
st.markdown("A unified platform comparing **Multilingual BERT (Encoder-Only)** classification and **Monolingual English GPT-2 (Decoder-Only)** generation.")
st.markdown("---")

# 4. Sidebar Controls
st.sidebar.header("⚙️ Architecture Controls")
model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    ("Multilingual BERT (Encoder-Only)", "Monolingual English GPT-2 (Decoder-Only)")
)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Project Overview:**\n\n"
    "• **BERT:** Multilingual classification model designed to categorize text sentiment (Positive, Negative, Neutral).\n\n"
    "• **GPT-2:** Monolingual English generative model designed for creative text completion and continuation."
)

# 5. Quick Sample Prompts
st.subheader("💡 Quick Test Prompts")
col1, col2, col3 = st.columns(3)

sample_text = ""
if col1.button("Positive 😊"):
    sample_text = "I absolutely love this product, it exceeded all my expectations!"
if col2.button("Negative 😞"):
    sample_text = "This is the worst experience I have ever had, totally disappointed."
if col3.button("Neutral ⚖️"):
    sample_text = "It is an okay product, nothing special, but gets the job done."

# 6. User Input Text Area
user_input = st.text_area(
    "✍️ Enter or modify your text below:", 
    value=sample_text,
    placeholder="Type a sentence in English, Urdu, or another supported language...",
    height=120
)

# 7. Execution Logic
st.markdown("")
if st.button("🚀 Run Model Inference"):
    if not user_input.strip():
        st.warning("⚠️ Please type or select some text first.")
    else:
        if model_choice == "Multilingual BERT (Encoder-Only)":
            with st.spinner("🔍 Loading Multilingual BERT & analyzing sentiment..."):
                try:
                    model_id = "Ahm-d18/my-bert-sentiment-model"
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForSequenceClassification.from_pretrained(model_id)
                    
                    inputs = tokenizer(user_input, return_tensors="pt")
                    outputs = model(**inputs)
                    prediction = torch.argmax(outputs.logits, dim=1).item()
                    
                    # Map prediction index to clean readable sentiment labels
                    labels = model.config.id2label if hasattr(model.config, "id2label") else {0: "Negative 😞", 1: "Neutral ⚖️", 2: "Positive 😊"}
                    sentiment_label = labels.get(prediction, f"Class {prediction}")
                    
                    st.success("✅ Multilingual Classification Complete!")
                    st.metric(label="Model Type", value="BERT (Encoder-Only)")
                    st.metric(label="Language Support", value="Multilingual")
                    st.metric(label="Predicted Sentiment", value=sentiment_label)
                    
                except Exception as e:
                    st.error(f"Error loading BERT model: {e}")
                    
        else:
            with st.spinner("🤖 Loading Monolingual GPT-2 & generating text..."):
                try:
                    model_id = "Ahm-d18/my-gpt2-sentiment"
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForCausalLM.from_pretrained(model_id)
                    
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                        
                    inputs = tokenizer(user_input, return_tensors="pt")
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=40, 
                        pad_token_id=tokenizer.eos_token_id
                    )
                    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    st.success("✅ Monolingual Generation Complete!")
                    st.metric(label="Model Type", value="GPT-2 (Decoder-Only)")
                    st.metric(label="Language Support", value="Monolingual (English)")
                    st.markdown("**Generated Text Continuation:**")
                    st.code(result, language="text")
                    
                except Exception as e:
                    st.error(f"Error loading GPT-2 model: {e}")
