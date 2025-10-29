
# ------------------------------
# 1. Base image
# ------------------------------
FROM python:3.10-slim

# ------------------------------
# 2. Set working directory
# ------------------------------
WORKDIR /app

# ------------------------------
# 3. Install system dependencies
# ------------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# 4. Copy project files
# ------------------------------
COPY . .

# ------------------------------
# 5. Install Python dependencies
# ------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------
# 6. Expose port
# ------------------------------
EXPOSE 5000

# ------------------------------
# 7. Run Flask app
# ------------------------------
CMD ["python", "server.py"]
