import uuid
import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import chromadb
from ollama import Client

app = FastAPI(
    title="Nextwork RAG API",
    description="A RAG (Retrieval-Augmented Generation) API using ChromaDB and Ollama",
    version="1.0.0"
)

# Initialize ChromaDB client and collection
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")

# Initialize Ollama client
# Parse OLLAMA_HOST - client expects hostname:port format, not URL
ollama_host_raw = os.getenv("OLLAMA_HOST", "localhost:11434")
# Remove protocol if present (http:// or https://)
ollama_host = ollama_host_raw.replace("http://", "").replace("https://", "")
ollama_client = Client(host=ollama_host)


class QueryRequest(BaseModel):
    q: str


class AddRequest(BaseModel):
    text: str


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Nextwork RAG API is running"}


@app.post("/add", status_code=status.HTTP_201_CREATED)
def add_knowledge(request: AddRequest):
    """Add new content to the knowledge base dynamically."""
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty"
        )
    
    try:
        # Generate a unique ID for this document
        doc_id = str(uuid.uuid4())
        
        # Add the text to Chroma collection
        collection.add(documents=[request.text], ids=[doc_id])
        
        return {
            "status": "success",
            "message": "Content added to knowledge base",
            "id": doc_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add content: {str(e)}"
        )


@app.post("/query")
def query(request: QueryRequest):
    """Query the knowledge base and get an AI-generated answer."""
    if not request.q or not request.q.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    try:
        # Query ChromaDB for relevant context
        results = collection.query(query_texts=[request.q], n_results=1)
        
        # Extract context from results
        context = ""
        if results["documents"] and len(results["documents"]) > 0:
            if len(results["documents"][0]) > 0:
                context = results["documents"][0][0]
        
        if not context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant context found in knowledge base"
            )
        
        # Generate answer using Ollama
        answer = ollama_client.generate(
            model="tinyllama",
            prompt=f"Context:\n{context}\n\nQuestion: {request.q}\n\nAnswer clearly and concisely:"
        )
        
        return {"answer": answer.response}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )
