FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY app.py embed.py k8s.txt ./
RUN pip install fastapi uvicorn chromadb ollama
RUN python embed.py
EXPOSE 8000
# Default OLLAMA_HOST for Docker (use host.docker.internal on Mac/Windows, or set via -e)
# Note: Ollama client expects hostname:port format, not URL
ENV OLLAMA_HOST=host.docker.internal:11434
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
