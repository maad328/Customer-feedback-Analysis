import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# --------------------------
# 1️⃣ Load model & tokenizer
# --------------------------
MODEL_DIR = "./bert_sentiment"

@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --------------------------
# 2️⃣ Streamlit UI
# --------------------------
st.set_page_config(page_title="BERT Sentiment Classifier", page_icon="💬")
st.title("💬 Sentiment Analysis (BERT)")

st.write(
    "Type any sentence below and the fine-tuned BERT model will classify it as **positive** or **negative**."
)

user_input = st.text_area("Enter text here:", height=150)

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # --------------------------
        # 3️⃣ Run model prediction
        # --------------------------
        encoded = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            predicted = torch.argmax(logits, dim=1).item()

        label_map = {0: "Negative 😞", 1: "Positive 😊"}
        st.subheader("Prediction:")
        st.success(label_map[predicted])
