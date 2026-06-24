import os
import streamlit as st
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

@st.cache_resource
def load_local_vector_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = load_local_vector_model()

def get_embedding(text):
    return model.encode(text)