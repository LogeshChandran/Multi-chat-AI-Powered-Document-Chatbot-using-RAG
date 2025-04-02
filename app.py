import os
import streamlit as st
import faiss
import torch
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import json


with open('config.json', 'r') as f:
    MODEL_CONFIG = json.load(f)
# ------------------------------- Set Streamlit Page Config ------------------------------- #
st.set_page_config(page_title="Chat with PDFs", layout="wide")

# Load Hugging Face Token
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ------------------------------- Custom CSS ------------------------------- #
st.markdown("""
    <style>
        .sidebar .sidebar-content {background-color: #f0f2f6;}
        .main > div {border-radius: 10px; padding: 15px;}
        div.stButton > button {width: 100%; border-radius: 10px; font-weight: bold;}
        .big-font {font-size: 20px !important;}
    </style>
""", unsafe_allow_html=True)

# ------------------------------- Model Classes ------------------------------- #

class TextEmbedder:
    def __init__(self, model_name="BAAI/bge-large-en"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)

class TextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text):
        return self.splitter.split_text(text)

class VectorDatabase:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add_embeddings(self, embeddings):
        self.index.add(embeddings.cpu().numpy())

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(query_embedding, top_k)
        return indices[0]

class Retriever:
    def __init__(self, texts, embeddings, vector_db):
        self.texts = texts
        self.vector_db = vector_db
        self.embeddings = embeddings
        self.bm25 = BM25Okapi([text.split() for text in texts])

    def retrieve(self, query, method="hybrid", top_k=5):
        query_embedding = TextEmbedder().embed_text([query]).cpu().numpy()
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        faiss_top_indices = self.vector_db.search(query_embedding, top_k)

        if method == "bm25":
            return [self.texts[i] for i in bm25_top_indices]
        elif method == "vector":
            return [self.texts[i] for i in faiss_top_indices]
        else:
            combined_scores = {i: bm25_scores[i] * 0.5 + (1 / (faiss_top_indices.tolist().index(i) + 1) * 0.5) if i in faiss_top_indices else bm25_scores[i] * 0.5 for i in range(len(self.texts))}
            final_indices = sorted(combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True)[:top_k]
            return [self.texts[i] for i in final_indices]

class LLM:
    def __init__(self, model_name, provider="huggingface"):
        self.provider = provider
        if provider == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                device_map="auto",
                use_auth_token=HUGGINGFACE_TOKEN
            )
        elif provider == "openai":
            import openai
            openai.api_key = OPENAI_API_KEY
            self.client = openai
            self.model_name = model_name
        elif provider == "gemini":
            from google.generativeai import configure, generate_text
            configure(api_key=GEMINI_API_KEY)
            self.generate_text = generate_text
            self.model_name = model_name
        elif provider == "llama":
            from llama_cpp import Llama
            self.model = Llama(model_path=model_name)

    def generate_response(self, query, context, temperature=0.7, top_p=0.9, top_k=50, max_tokens=150):
        if self.provider == "huggingface":
            prompt = f"Context:\n{context}\n\nUser: {query}\n\nAssistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        elif self.provider == "openai":
            response = self.client.Completion.create(
                model=self.model_name,
                prompt=f"Context:\n{context}\n\nUser: {query}\n\nAssistant:",
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
            return response.choices[0].text.strip()
        elif self.provider == "gemini":
            response = self.generate_text(model=self.model_name, prompt=f"Context:\n{context}\n\nUser: {query}\n\nAssistant:")
            return response["candidates"][0]["content"]
        elif self.provider == "llama":
            output = self.model(f"Context:\n{context}\n\nUser: {query}\n\nAssistant:", max_tokens=max_tokens)
            return output["choices"][0]["text"]

# ------------------------------- Streamlit App ------------------------------- #

if "chats" not in st.session_state:
    st.session_state.chats = []
if "chat_data" not in st.session_state:
    st.session_state.chat_data = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "retrieval_method" not in st.session_state:
    st.session_state.retrieval_method = "hybrid"

# Layout: Left (Chat List) | Middle (Chat) | Right (Configuration)
col1, col2, col3 = st.columns([0.6, 4, 0.8])

# ------------------------------- Left Sidebar - Chat Sessions ------------------------------- #
with col1:
    st.subheader("📂 Chat Sessions")

    if st.button("+ New Chat", key="new_chat"):
        chat_id = f"chat_{len(st.session_state.chats) + 1}"
        st.session_state.chats.append(chat_id)
        st.session_state.chat_data[chat_id] = None
        st.session_state.current_chat = chat_id

    for chat_id in st.session_state.chats:
        chat_name = f"Chat {st.session_state.chats.index(chat_id) + 1}"
        
        colA, colB, colC = st.columns([6, 2, 2])
        
        if colA.button(chat_name, key=f"select_{chat_id}"):
            st.session_state.current_chat = chat_id

        if colB.button("✏️", key=f"edit_{chat_id}"):
            st.session_state.chat_data[chat_id] = "Editing"

        if colC.button("❌", key=f"delete_{chat_id}"):
            st.session_state.chats.remove(chat_id)
            del st.session_state.chat_data[chat_id]
            st.session_state.current_chat = None if len(st.session_state.chats) == 0 else st.session_state.chats[-1]
            st.rerun()

# ------------------------------- Middle Section - PDF Upload & Chat ------------------------------- #
with col2:
    st.subheader("📜 Upload PDF & Chat")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        # Enable process button once PDF is uploaded
        if st.button("Process PDF"):
            pdf_reader = PdfReader(uploaded_file)
            pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            chunks = TextSplitter().split_text(pdf_text)
            embeddings = TextEmbedder().embed_text(chunks)
            vector_db = VectorDatabase(embedding_dim=embeddings.shape[1])
            vector_db.add_embeddings(embeddings)
            st.session_state.chat_data[st.session_state.current_chat] = (chunks, embeddings, vector_db)
            st.session_state.chat_data[st.session_state.current_chat]["processed"] = True

    if st.session_state.current_chat and "processed" in st.session_state.chat_data[st.session_state.current_chat]:
        user_query = st.text_input("Ask a question:", key=f"query_{st.session_state.current_chat}")

        if user_query:
            retrieved_docs = Retriever(*st.session_state.chat_data[st.session_state.current_chat]).retrieve(user_query, method=st.session_state.retrieval_method)
            response = LLM(provider=st.session_state.model_provider, model_name=st.session_state.model_name).generate_response(user_query, "\n".join(retrieved_docs))
            st.write(f"**AI:** {response}")

# ------------------------------- Right Sidebar - Configuration Panel ------------------------------- #
with col3:
    st.subheader("⚙️ Configuration")

    # --- Text Embedding Model --- #
    embedding_model = st.selectbox("Text Embedding Model", ["BAAI/bge-large-en", "sentence-transformers/all-MiniLM-L6-v2"], index=0)
    st.session_state.embedding_model = embedding_model

    # --- Text Splitter --- #
    chunk_size = st.slider("Text Splitter Chunk Size", 100, 1000, 500, 100)
    chunk_overlap = st.slider("Text Splitter Chunk Overlap", 50, 500, 100, 50)
    st.session_state.chunk_size = chunk_size
    st.session_state.chunk_overlap = chunk_overlap

    # --- Model Provider --- #
    model_provider = st.selectbox("Model Provider", MODEL_CONFIG.keys(), index=0)
    st.session_state.model_provider = model_provider

    model_names = MODEL_CONFIG.get(model_provider, {}).get("models", [])
    model_name = st.selectbox("Model Name", model_names, index=0)
    st.session_state.model_name = model_name

    # --- Retriever Algorithm --- #
    retrieval_method = st.selectbox("Retriever Algorithm", ["hybrid", "bm25", "vector"], index=0)
    st.session_state.retrieval_method = retrieval_method

    # --- LLM Hyperparameters --- #
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    top_p = st.slider("Top-P", 0.0, 1.0, 0.9, 0.05)
    top_k = st.slider("Top-K", 1, 100, 50, 1)
    max_tokens = st.slider("Max Tokens", 50, 4096, 150, 50)
    st.session_state.temperature = temperature
    st.session_state.top_p = top_p
    st.session_state.top_k = top_k
    st.session_state.max_tokens = max_tokens

    if st.button("Update Settings"):
        st.rerun()
