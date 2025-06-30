import streamlit as st
import subprocess
import tempfile
import os
import uuid


st.set_page_config(page_title="PDF to Markdown", layout="wide")
st.title("📄 PDF to Markdown Converter")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        tmp_pdf.flush()

        # Run the pdf2md command via subprocess
        try:
            with st.spinner("Converting to Markdown..."):
                result = subprocess.run(
                    ["npx", "pdf2md", tmp_pdf.name, f"project_{uuid.uuid4().hex[:8]}"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    markdown_output = result.stdout
                    st.subheader("✅ Markdown Output")
                    st.code(markdown_output, language="markdown")

                    st.download_button(
                        label="⬇️ Download Markdown",
                        data=markdown_output,
                        file_name="converted.md",
                        mime="text/markdown"
                    )
                else:
                    st.error("Conversion failed. Check error log below:")
                    st.code(result.stderr)

        except subprocess.TimeoutExpired:
            st.error("Conversion timed out. Try a smaller PDF.")
        except Exception as e:
            st.exception(f"Unexpected error: {e}")

    # Clean up temp file
    if os.path.exists(tmp_pdf.name):
        os.unlink(tmp_pdf.name)

