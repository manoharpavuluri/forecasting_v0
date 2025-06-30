import streamlit as st
import os

st.title("Simple File Uploader Test")

uploaded_file = st.file_uploader("Upload any file here", type=["parquet", "csv", "txt"])

if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.write("File content (first 100 bytes):")
        st.code(uploaded_file.getvalue()[:100])
else:
        st.info("Please upload a file.")

st.write(f"DEBUG: uploaded_file is {uploaded_file}")