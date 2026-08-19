FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EMBEDDING_DEVICE=cpu \
    PROJECT_ROOT=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src
# Installing the CPU wheel first prevents pip from pulling unnecessary CUDA
# runtimes into a CPU-only deployment. The second command sees it as satisfied.
RUN pip install --no-cache-dir "torch>=2.2" \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps .

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/mamlaka_ai/ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
