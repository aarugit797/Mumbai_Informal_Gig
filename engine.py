import os
import torch

# 1. Hide the GPU from the system environment
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 2. Force torch to report that no GPU is available
torch.cuda.is_available = lambda : False

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 2. DIRECTORIES ---
DATA_DIR = "./data"
PERSIST_DIR = "./mumbai_worker_db"
EMBEDDING_MODEL = "BAAI/bge-m3"

# --- 3. RETRIEVERS (FORCING CPU) ---
# Note: Changed 'device': 'cuda' to 'cpu'
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'}  
)

# Load existing vectorstore
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
dense_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 7})

# BM25 setup
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

# --- 4. LLM SETUP (CPU ONLY) ---
model_id = "google/gemma-2-2b-it"

# Removed quantization_config and load_in_4bit
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.7)
llm = HuggingFacePipeline(pipeline=pipe)

# --- 5. RAG CHAIN ---
prompt = PromptTemplate.from_template("""You are a helpful assistant for gig and informal workers in Mumbai.
Use ONLY the retrieved information.
Answer in the same language as the question.

Context: {context}
Question: {question}

Answer:
{sources}
Disclaimer: यह केवल सामान्य जानकारी है। यह कानूनी सलाह नहीं है।""")

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