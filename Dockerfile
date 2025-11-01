
# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- Set Workdir ----------
WORKDIR /app

# ---------- Install system dependencies ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ---------- Copy project ----------
COPY requirements.txt .
COPY . .

# ---------- Environment ----------
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---------- Install lightweight CPU PyTorch ----------
RUN pip install torch==2.3.0+cpu torchvision==0.18.0+cpu torchaudio==2.3.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ---------- Install other dependencies ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Create output directories (optional but safe) ----------
RUN mkdir -p uploads outputs

# ---------- Expose port & run ----------
EXPOSE 8080

# Using Gunicorn for production (4 workers recommended for small EC2)
CMD ["gunicorn", "--workers", "4", "--threads", "2", "--bind", "0.0.0.0:8080", "server:app"]
