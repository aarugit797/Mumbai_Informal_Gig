import os
import torch

# 1. Hide the GPU from the system environment
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 2. Force torch to report that no GPU is available
torch.cuda.is_available = lambda : False

import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION & SECRETS ---
try:
    # This pulls from the Streamlit Cloud Dashboard, NOT this file
    HF_TOKEN = st.secrets["HF_TOKEN"]
except KeyError:
    st.error("Please add HF_TOKEN to your Streamlit Secrets dashboard!")
    st.stop()

DATA_DIR = "./data"
PERSIST_DIR = "./mumbai_worker_db"
EMBEDDING_MODEL = "BAAI/bge-m3"

# --- 2. RETRIEVERS ---
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'} 
)

vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
dense_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 7})

loader = PyPDFDirectoryLoader(DATA_DIR)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 7

ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.65, 0.35]
)

# --- 3. LLM SETUP (API VERSION) ---
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

# --- 4. RAG CHAIN ---
prompt = PromptTemplate.from_template("""You are a helpful assistant for workers in Mumbai.
Context: {context}
Question: {question}
Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_sources(docs):
    return ", ".join(set(os.path.basename(doc.metadata.get("source", "Unknown")) for doc in docs))

rag_chain = (
    {
        "context": ensemble_retriever | format_docs,
        "question": RunnablePassthrough(),
        "sources": ensemble_retriever | format_sources  
    }
    | prompt | llm | StrOutputParser()
)