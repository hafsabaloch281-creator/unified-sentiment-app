import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch

# App Configuration
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🌟", layout="centered")

st.title("🌟 Sentiment Analysis: BERT vs. GPT-2")
st.write("Welcome! This app lets you test and compare sentiment analysis using your two fine-tuned Hugging Face models.")

# Sidebar Model Selection
model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    ("Encoder-Only (BERT)", "Decoder-Only (GPT-2)")
)

# User Input Text Area
user_input = st.text_area(
    "Enter your text to analyze sentiment:", 
    placeholder="Type a sentence like 'I am really happy with this experience!'..."
)

# Execution Button
if st.button("Run Analysis"):
    if not user_input.strip():
        st.warning("Please type some text first.")
    else:
        if model_choice == "Encoder-Only (BERT)":
            with st.spinner("Loading BERT model and analyzing..."):
                try:
                    model_id = "Ahm-d18/my-bert-sentiment-model"
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForSequenceClassification.from_pretrained(model_id)
                    
                    inputs = tokenizer(user_input, return_tensors="pt")
                    outputs = model(**inputs)
                    prediction = torch.argmax(outputs.logits, dim=1).item()
                    
                    st.success("Analysis Complete!")
                    st.write(f"**Selected Model:** BERT (Encoder-Only)")
                    st.write(f"**Predicted Sentiment Class Index:** `{prediction}`")
                except Exception as e:
                    st.error(f"Error loading or running BERT model: {e}")
                    
        else:
            with st.spinner("Loading GPT-2 model and generating..."):
                try:
                    model_id = "Ahm-d18/my-gpt2-sentiment"
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForCausalLM.from_pretrained(model_id)
                    
                    # Ensure padding token is handled for GPT-2
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                        
                    inputs = tokenizer(user_input, return_tensors="pt")
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=30, 
                        pad_token_id=tokenizer.eos_token_id
                    )
                    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    st.success("Analysis Complete!")
                    st.write(f"**Selected Model:** GPT-2 (Decoder-Only)")
                    st.code(result)
                except Exception as e:
                    st.error(f"Error loading or running GPT-2 model: {e}")
