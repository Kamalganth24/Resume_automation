# app.py
import streamlit as st
import numpy as np
import sys
from main import control_flow
from embed import model

@st.cache_resource
def start_monitor():
    import subprocess
    subprocess.Popen([sys.executable, "monitor.py"])

start_monitor()

@st.cache_resource
def get_index_and_entries():
    # This will only run once unless inputs change
    index, entries = control_flow()
    return index, entries

# Build FAISS index and chunks (cached)
index, entries = get_index_and_entries()

st.title("Resume Semantic Search")

query = st.text_input("Enter your search query:")

if query:
    query_embedding = model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    distances, indices = index.search(query_embedding.astype("float32"), k=3)

    st.subheader("Top Matches")
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(entries):
            st.write(f"**Result {i+1}:**")
            st.write(entries[idx])  # show the resume chunk
        st.write(f"Similarity Score: {distances[0][i]:.3f}")

