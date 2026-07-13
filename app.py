import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
# Bulletproof updated imports:
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile

st.set_page_config(page_title="Synthesis PoC (Free Tier)", page_icon="🔮", layout="wide")

st.title("🔮 Synthesis: Multi-Source Knowledge Base")
st.write("Upload multiple PDFs or text documents to create a unified blueprint and cross-reference data for free.")

# Automatically fetch the free Groq key from Streamlit Secrets backend
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("Missing GROQ_API_KEY! Please add it to your Streamlit App Secrets.")
    st.stop()

# Initialize session state for the vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# File Uploader
uploaded_files = st.file_uploader(
    "Drop your source files here (PDF or TXT)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process & Synthesize Sources", type="primary"):
        all_docs = []
        
        with st.spinner("Parsing and cross-referencing documents..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                
                try:
                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(temp_file_path)
                        docs = loader.load()
                    else:
                        with open(temp_file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        from langchain_core.documents import Document
                        docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]
                    
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                    all_docs.extend(docs)
                finally:
                    os.unlink(temp_file_path)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)
            
            # Using a free, locally running embedding model in the cloud container (no API key needed)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vector_store = Chroma.from_documents(splits, embeddings)
            
        st.success(f"Successfully synthesized {len(uploaded_files)} sources into a unified knowledge base!")

# Query / Chat Interface
st.markdown("---")
st.subheader("Query the Synthesis Blueprint")

if st.session_state.vector_store is not None:
    user_query = st.text_input("Ask a question that spans across your uploaded sources:")
    
    if user_query:
        with st.spinner("Analyzing cross-references..."):
            # Using the fast, free Llama 3.3 model via Groq
            llm = ChatGroq(model="llama-3.3-70b-specdec", groq_api_key=groq_api_key, temperature=0.2)
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
            
            system_prompt = (
                "You are the engine of Synthesis, an advanced multi-source document aggregator.\n"
                "Analyze the provided context pulled from different files. Connect overlapping data points, "
                "resolve conflicts gracefully, and synthesize a clear, unified answer.\n"
                "Always cite the source filenames when presenting data.\n\n"
                "Context:\n{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
       from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
            
            response = rag_chain.invoke({"input": user_query})
            
            st.markdown("### Synthesized Answer")
            st.write(response["answer"])
            
            with st.expander("View Source Snippets Used for this Answer"):
                for doc in response["context"]:
                    st.markdown(f"**Source:** `{doc.metadata.get('source', 'Unknown')}`")
                    st.caption(doc.page_content)
                    st.markdown("---")
else:
    st.info("Upload and process documents above to unlock the synthesis chat interface.")
