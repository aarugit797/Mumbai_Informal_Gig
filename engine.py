import os
import torch
import sys

# 1. Hide the GPU from the system environment
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 2. Force torch to report that no GPU is available
torch.cuda.is_available = lambda : False

# 3. Handle sqlite3 version for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
DATA_DIR = "./data"
PERSIST_DIR = "./mumbai_worker_db"
EMBEDDING_MODEL = "BAAI/bge-m3"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_sources(docs):
    return ", ".join(set(os.path.basename(doc.metadata.get("source", "Unknown")) for doc in docs))

@st.cache_resource
def get_rag_chain():
    try:
        HF_TOKEN = st.secrets["HF_TOKEN"]
    except KeyError:
        st.error("Please add HF_TOKEN to your Streamlit Secrets dashboard!")
        st.stop()

    # --- 1. EMBEDDINGS & VECTORSTORE ---
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=HF_TOKEN
    )
    
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    dense_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 7})

    # --- 2. BM25 RETRIEVER (requires loading documents) ---
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 7

    # --- 3. ENSEMBLE RETRIEVER ---
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.65, 0.35]
    )

    # --- 4. LLM SETUP ---
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-2b-it",
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )

    prompt = PromptTemplate.from_template("""You are a helpful assistant for workers in Mumbai.
Context: {context}
Question: {question}
Answer:""")

    # Build Chain
    chain = (
        {
            "context": ensemble_retriever | format_docs,
            "question": RunnablePassthrough(),
            "sources": ensemble_retriever | format_sources  
        }
        | prompt | llm | StrOutputParser()
    )
    
    return chain

# Expose the chain for app.py
rag_chain = get_rag_chain()