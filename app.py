import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="Synthesis PoC (Free Tier)", page_icon="🔮", layout="wide")

st.title("🔮 Synthesis: Multi-Source Knowledge Base")
st.write(
    "Upload multiple PDFs or text documents to create a unified blueprint "
    "and cross-reference data for free."
)

# --- API key from Streamlit Secrets only. Never hardcode it here. ---
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("Missing GROQ_API_KEY! Add it under your Streamlit app's Settings → Secrets.")
    st.stop()

# --- Session state ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- File uploader ---
uploaded_files = st.file_uploader(
    "Drop your source files here (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("Process & Synthesize Sources", type="primary"):
        all_docs = []

        with st.spinner("Parsing and cross-referencing documents..."):
            for uploaded_file in uploaded_files:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                try:
                    if uploaded_file.name.lower().endswith(".pdf"):
                        loader = PyPDFLoader(temp_file_path)
                        docs = loader.load()
                    else:
                        with open(temp_file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]

                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                    all_docs.extend(docs)
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
                finally:
                    os.unlink(temp_file_path)

            if not all_docs:
                st.warning("No content could be extracted from the uploaded files.")
                st.stop()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vector_store = Chroma.from_documents(splits, embeddings)

        st.success(f"Successfully synthesized {len(uploaded_files)} source(s) into a unified knowledge base!")

# --- Query interface ---
st.markdown("---")
st.subheader("Query the Synthesis Blueprint")

if st.session_state.vector_store is not None:
    user_query = st.text_input("Ask a question that spans across your uploaded sources:")

    if user_query:
        with st.spinner("Analyzing cross-references..."):
            llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key, temperature=0.2)
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})

            system_prompt = (
                "You are the engine of Synthesis, an advanced multi-source document aggregator.\n"
                "Analyze the provided context pulled from different files. Connect overlapping data "
                "points, resolve conflicts gracefully, and synthesize a clear, unified answer.\n"
                "Always cite the source filenames when presenting data.\n\n"
                "Context:\n{context}"
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            try:
                response = rag_chain.invoke({"input": user_query})
            except Exception as e:
                st.error(f"Error while generating answer: {type(e).__name__}: {e}")
                st.exception(e)
                st.stop()

            st.markdown("### Synthesized Answer")
            st.write(response["answer"])

            with st.expander("View Source Snippets Used for this Answer"):
                for doc in response["context"]:
                    st.markdown(f"**Source:** `{doc.metadata.get('source', 'Unknown')}`")
                    st.caption(doc.page_content)
                    st.markdown("---")
else:
    st.info("Upload and process documents above to unlock the synthesis chat interface.")
