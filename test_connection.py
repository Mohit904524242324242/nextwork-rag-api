#!/usr/bin/env python3
"""Test script to verify Ollama and ChromaDB connections."""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_ollama():
    """Test Ollama connection."""
    print("Testing Ollama connection...")
    try:
        from ollama import Client
        ollama_host_raw = os.getenv("OLLAMA_HOST", "localhost:11434")
        ollama_host = ollama_host_raw.replace("http://", "").replace("https://", "")
        print(f"  Using host: {ollama_host}")
        
        client = Client(host=ollama_host)
        result = client.generate(model="tinyllama", prompt="Hello")
        print(f"  ✓ Ollama connection successful")
        print(f"  ✓ Response type: {type(result)}")
        print(f"  ✓ Response attribute: {hasattr(result, 'response')}")
        print(f"  ✓ Sample response: {result.response[:50]}...")
        return True
    except Exception as e:
        print(f"  ✗ Ollama connection failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chromadb():
    """Test ChromaDB connection."""
    print("\nTesting ChromaDB connection...")
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./db")
        collection = client.get_or_create_collection("docs")
        count = collection.count()
        print(f"  ✓ ChromaDB connection successful")
        print(f"  ✓ Collection 'docs' has {count} documents")
        return True
    except Exception as e:
        print(f"  ✗ ChromaDB connection failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app():
    """Test the app endpoints."""
    print("\nTesting app endpoints...")
    try:
        from app import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test health check
        response = client.get("/")
        print(f"  ✓ Health check: {response.json()}")
        
        # Test add
        response = client.post("/add", json={"text": "Test document for connection testing."})
        print(f"  ✓ Add endpoint: {response.status_code}")
        
        # Test query
        response = client.post("/query", json={"q": "What is this test about?"})
        if response.status_code == 200:
            print(f"  ✓ Query endpoint: {response.status_code}")
            print(f"  ✓ Response: {response.json()['answer'][:50]}...")
        else:
            print(f"  ✗ Query endpoint failed: {response.status_code}")
            print(f"  ✗ Response: {response.text}")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ App test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Nextwork RAG API Connection Test")
    print("=" * 60)
    
    ollama_ok = test_ollama()
    chroma_ok = test_chromadb()
    app_ok = test_app()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Ollama:  {'✓ OK' if ollama_ok else '✗ FAILED'}")
    print(f"  ChromaDB: {'✓ OK' if chroma_ok else '✗ FAILED'}")
    print(f"  App:     {'✓ OK' if app_ok else '✗ FAILED'}")
    print("=" * 60)
    
    if all([ollama_ok, chroma_ok, app_ok]):
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
