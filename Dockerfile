FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    INTERVAL_PUBLIC_DEMO=1 INTERVAL_RUNTIME=/tmp/interval-runtime \
    HF_HOME=/tmp/huggingface OMP_NUM_THREADS=2
WORKDIR /app
COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-docker.txt
COPY . /app
RUN pip install --no-deps -e . && python scripts/download_model.py
RUN useradd -m -u 1000 user
USER user
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "interval.server:app", "--host", "0.0.0.0", "--port", "7860"]
