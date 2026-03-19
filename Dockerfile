FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY ui/ ui/
COPY scripts/ scripts/
COPY tests/fixtures/ tests/fixtures/
COPY .env.example .env.example

# Create data directories
RUN mkdir -p data/chroma data/uploads

# Expose ports
EXPOSE 8000 8501

# Default: run FastAPI backend
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
