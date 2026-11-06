import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch

# 1. Page Configuration (Wide layout for a modern dashboard look)
st.set_page_config(
    page_title="AI Sentiment Hub", 
    page_icon="✨", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling buttons, containers, and typography
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Header Section
st.title("✨ AI Sentiment Analysis Hub")
st.markdown("Compare predictions and text generation between your **BERT (Encoder-Only)** and **GPT-2 (Decoder-Only)** fine-tuned models.")
st.markdown("---")

# 3. Sidebar Controls
st.sidebar.header("⚙️ Model Controls")
model_choice = st.sidebar.selectbox(
    "Choose Model Architecture",
    ("Encoder-Only (BERT)", "Decoder-Only (GPT-2)")
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Models Info:**\n"
    "- **BERT:** Best for classification (Positive/Negative).\n"
    "- **GPT-2:** Best for text continuation & generation."
)

# 4. Quick Sample Prompts (Interactive feature)
st.subheader("💡 Try Sample Text")
col1, col2, col3 = st.columns(3)

sample_text = ""
if col1.button("Positive 😊"):
    sample_text = "I absolutely love this product, it exceeded all my expectations!"
if col2.button("Negative 😞"):
    sample_text = "This is the worst experience I have ever had, totally disappointed."
if col3.button("Neutral ⚖️"):
    sample_text = "It is an okay product, nothing special, but gets the job done."

# 5. User Input Text Area (Pre-filled if a sample button is clicked)
user_input = st.text_area(
    "✍️ Enter or modify your text below:", 
    value=sample_text,
    placeholder="Type a sentence here...",
    height=120
)

# 6. Execution Button & Logic
st.markdown("")
if st.button("🚀 Run Sentiment Analysis"):
    if not user_input.strip():
        st.warning("⚠️ Please type or select some text first.")
    else:
        if model_choice == "Encoder-Only (BERT)":
            with st.spinner("🔍 Loading BERT & analyzing sentiment..."):
                try:
                    model_id = "Ahm-d18/my-bert-sentiment-model"
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForSequenceClassification.from_pretrained(model_id)
                    
                    inputs = tokenizer(user_input, return_tensors="pt")
                    outputs = model(**inputs)
                    prediction = torch.argmax(outputs.logits, dim=1).item()
                    
                    # Display results in clean metric boxes
                    st.success("✅ Analysis Complete!")
                    st.metric(label="Selected Architecture", value="BERT (Encoder-Only)")
                    st.metric(label="Predicted Sentiment Class Index", value=prediction)
                except Exception as e:
                    st.error(f"Error loading BERT model: {e}")
                    
        else:
            with st.spinner("🤖 Loading GPT-2 & generating response..."):
                try:
                    model_id = "Ahm-d18/my-gpt2-sentiment"
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForCausalLM.from_pretrained(model_id)
                    
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                        
                    inputs = tokenizer(user_input, return_tensors="pt")
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=35, 
                        pad_token_id=tokenizer.eos_token_id
                    )
                    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    st.success("✅ Generation Complete!")
                    st.metric(label="Selected Architecture", value="GPT-2 (Decoder-Only)")
                    st.markdown("**Model Output / Continuation:**")
                    st.code(result, language="text")
                except Exception as e:
                    st.error(f"Error loading GPT-2 model: {e}")
