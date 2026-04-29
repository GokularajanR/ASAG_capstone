FROM python:3.13-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer cache: only re-installs if these change)
COPY pyproject.toml uv.lock ./

# Install all dependencies into the system Python (no venv inside container)
RUN uv sync --frozen --no-dev

# Pre-download NLTK data so the first request isn't slow
RUN uv run python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Copy the rest of the source
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY main.py ./
COPY grade_mapper.joblib ./

# data/ and grade_mapper.joblib are NOT copied — they come from the Azure Files mount at runtime
# If running locally without a mount, create an empty data dir so the store doesn't crash on startup
RUN mkdir -p data

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
