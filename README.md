# Multi-chat-AI-Powered-Document-Chatbot-using-RAG
Multi chat AI-Powered Document Chatbot using RAG

## Overview
The **Chat with PDFs** application enables users to upload PDF files, extract text, split the text into smaller chunks, embed the chunks into a vector database (FAISS), and retrieve relevant sections using a combination of BM25 and vector search. The retrieved text is then passed to an LLM (Large Language Model) to generate AI-powered responses.

## Features
- **PDF Upload & Processing**: Extracts text from PDF files.
- **Text Splitting**: Splits extracted text into smaller chunks.
- **Text Embedding**: Converts text chunks into embeddings using a SentenceTransformer model.
- **Vector Search (FAISS)**: Enables semantic search over embedded text.
- **BM25 Retrieval**: Uses BM25 for keyword-based retrieval.
- **Hybrid Search**: Combines BM25 and FAISS results.
- **LLM Integration**: Supports multiple LLM providers (Hugging Face, OpenAI, Gemini, and Llama).
- **Configurable Hyperparameters**: Users can adjust settings such as chunk size, retrieval method, and LLM parameters.

## Dependencies
```bash
pip install streamlit faiss-cpu torch PyPDF2 sentence-transformers rank-bm25 transformers python-dotenv
```

## File Structure
```
/ChatWithPDFs
|-- app.py               # Main application file
|-- config.json          # Configuration for LLM models
|-- .env                 # Environment variables (Hugging Face, OpenAI, Gemini API Keys)
```

## Configuration
**config.json** contains model provider and model name mappings. Example:
```json
{
    "huggingface": {"models": ["mistralai/Mistral-7B-Instruct", "meta-llama/Llama-2-7b-chat-hf"]},
    "openai": {"models": ["gpt-3.5-turbo", "gpt-4"]},
    "gemini": {"models": ["gemini-pro"]},
    "llama": {"models": ["path/to/local/llama/model"]}
}
```

## Environment Variables
Store API keys in a `.env` file:
```
HUGGINGFACE_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Components

### 1. **TextEmbedder**
Encodes text into vector embeddings using a SentenceTransformer model.
```python
class TextEmbedder:
    def __init__(self, model_name="BAAI/bge-large-en"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)
```

### 2. **TextSplitter**
Splits extracted text into smaller chunks.
```python
class TextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text):
        return self.splitter.split_text(text)
```

### 3. **VectorDatabase**
Stores and retrieves embeddings using FAISS.
```python
class VectorDatabase:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add_embeddings(self, embeddings):
        self.index.add(embeddings.cpu().numpy())

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(query_embedding, top_k)
        return indices[0]
```

### 4. **Retriever**
Retrieves relevant text chunks using BM25, FAISS, or a hybrid approach.
```python
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
```

### 5. **LLM**
Handles interactions with various language models.
```python
class LLM:
    def __init__(self, model_name, provider="huggingface"):
        self.provider = provider
        if provider == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", use_auth_token=HUGGINGFACE_TOKEN)
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
```

## Streamlit App Workflow
1. **Upload PDF** → Extract text.
2. **Process PDF** → Split text into chunks and embed them.
3. **User Queries** → Retrieve relevant chunks using BM25/FAISS.
4. **Generate AI Response** → Use LLM to answer the user.

## Configuration Panel
Users can configure:
- **Embedding Model**
- **Chunk Size & Overlap**
- **Model Provider & Model Name**
- **Retrieval Method** (Hybrid, BM25, FAISS)
- **LLM Hyperparameters** (Temperature, Top-P, Top-K, Max Tokens)

## Conclusion
This application allows users to efficiently extract and retrieve relevant information from PDFs while leveraging state-of-the-art LLMs for AI-generated responses.

![image](https://github.com/user-attachments/assets/439434ad-7c8b-40eb-80be-63cfc7699cac)

![image](https://github.com/user-attachments/assets/cecaae52-5ff1-445d-ab22-c57d3e4efebc)
