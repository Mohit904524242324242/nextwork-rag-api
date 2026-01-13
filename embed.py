"""Script to embed documents into ChromaDB for the RAG API."""
import chromadb
import os
import sys


def embed_file(file_path: str, doc_id: str = None):
    """
    Embed a text file into ChromaDB.
    
    Args:
        file_path: Path to the text file to embed
        doc_id: Optional document ID. If not provided, uses the filename without extension
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Create client and collection
    client = chromadb.PersistentClient(path="./db")
    collection = client.get_or_create_collection("docs")
    
    # Read the text file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    if not text.strip():
        raise ValueError(f"File {file_path} is empty")
    
    # Use filename as ID if not provided
    if doc_id is None:
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
    
    # Add document to collection
    # ChromaDB will automatically generate embeddings using the default embedding function
    collection.add(documents=[text], ids=[doc_id])
    
    print(f"✓ Successfully embedded '{file_path}' with ID '{doc_id}' into ChromaDB")
    return doc_id


if __name__ == "__main__":
    # Default to k8s.txt if no argument provided
    file_path = sys.argv[1] if len(sys.argv) > 1 else "k8s.txt"
    
    try:
        embed_file(file_path)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
