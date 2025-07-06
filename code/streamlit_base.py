import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.pipelines import pipeline
import torch
import pandas as pd
import json
import re
from datetime import datetime

# --- Streamlit Config ---
st.set_page_config(page_title="Mistral Domain Name Generator", layout="centered")
st.title("🌱 Creative Domain Name Generator (Mistral-7B)")

# --- Sidebar Settings ---
with st.sidebar:
    st.write("Powered by Mistral-7B-Instruct v0.3 (4-bit)")
    temperature = 0.7
    top_k = 50
    top_p = 0.90
    max_tokens = 350

# --- Model Loading ---
@st.cache_resource
def load_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    return pipe

pipe = load_model()

# --- Session State Setup ---
if "all_suggestions" not in st.session_state:
    st.session_state.all_suggestions = []

# --- Input ---
user_prompt = st.text_input("Enter a business description:", "vegan food delivery service")

# --- Generate Button ---
if st.button("Generate Domain Names"):
    prompt = f"""
Suggest 10 creative and brandable .com domain names for a business described as:
"{user_prompt}"

Return only a plain JSON array of strings, like:
[
  "veggiedelivered.com",
  "plantpoweredmeals.com"
]

Only return valid JSON — no explanation or formatting.
"""

    with st.spinner("Thinking..."):
        response = pipe(prompt)
        generated = response[0]["generated_text"].replace(prompt, "").strip()

        # Extract the first valid JSON list of strings
        try:
            match = re.search(r"\[\s*\".*?\"\s*\]", generated, re.DOTALL)
            if match:
                suggestions_list = json.loads(match.group())
            else:
                raise ValueError("No valid JSON array found.")
        except Exception as e:
            st.error("❌ Failed to parse response as JSON. Try again or refine prompt.")
            st.text_area("🔍 Raw Model Output (Debug)", generated, height=200)
            suggestions_list = []

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        batch_data = [
            {
                "domain": domain.strip(),
                "timestamp": timestamp,
                "description": user_prompt
            }
            for domain in suggestions_list if domain.strip()
        ]

        st.session_state.all_suggestions.extend(batch_data)

        # Save to CSV
        df = pd.DataFrame(st.session_state.all_suggestions)
        df.to_csv("all_suggestions.csv", index=False)

# --- Show All Suggestions So Far ---
if st.session_state.all_suggestions:
    st.markdown("### 🗂️ All Generated Suggestions")
    df_all = pd.DataFrame(st.session_state.all_suggestions)
    st.dataframe(df_all)

    # --- Download Button ---
    csv = df_all.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download All as CSV", csv, "domain_suggestions.csv", "text/csv")
