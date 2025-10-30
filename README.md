# YOLO + OCR + Background Removal API

A Flask-based REST API for image processing tasks including object detection (YOLO), background removal, and optical character recognition (OCR). This server processes images and returns results with persistent storage for each user.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Response Format](#response-format)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)

## Features

- **Object Detection**: Identify and locate objects in images using YOLO v8 model
- **Background Removal**: Remove backgrounds from images using AI-powered segmentation
- **Text Extraction**: Extract text from images using OCR (Tesseract)
- **User-Based Storage**: Keep separate output folders for each user
- **CORS Support**: Cross-origin requests enabled for web clients
- **Error Handling**: Comprehensive error messages and automatic cleanup

## Prerequisites

### System Requirements

- Python 3.8 or higher
- For OCR: Tesseract-OCR installed on your system
- For GPU acceleration: NVIDIA GPU with CUDA support (optional, for faster detection)
- Disk space for model files and outputs (~1GB+ recommended)

### Required Python Libraries

```
Flask
Flask-CORS
Werkzeug
ultralytics (YOLOv8)
rembg (Background removal)
Pillow (PIL)
pytesseract
```

### System-Specific Setup

**Windows:**
1. Download Tesseract-OCR installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install to `C:\Program Files\Tesseract-OCR\`
3. Uncomment in code: `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`

**Linux/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
# Path will be: /usr/bin/tesseract
```

**EC2/Docker:**
```bash
sudo apt-get install tesseract-ocr
# Uncomment in code: pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
```

## Installation

### Step 1: Clone or Create Project

```bash
mkdir yolo-api && cd yolo-api
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install flask flask-cors werkzeug ultralytics rembg pillow pytesseract
```

For GPU support (NVIDIA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Download YOLO Model

The server requires a trained YOLO model file named `object_identify.pt`. Place it in the project root:

```bash
# Option 1: Use a pre-trained YOLOv8 model
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.save('object_identify.pt')"

# Option 2: Use your custom trained model
# Copy your trained model as: object_identify.pt
```

## Configuration

### Model Path

Edit the `MODEL_PATH` variable in the code:
```python
MODEL_PATH = "object_identify.pt"  # Ensure this file exists
```

### Upload/Output Directories

Configure paths in the configuration section:
```python
UPLOAD_DIR = "uploads"      # Temporary storage for uploaded images
OUTPUT_DIR = "outputs"      # Persistent storage for processed outputs
ALLOWED = {'jpg', 'jpeg', 'png'}  # Allowed file types
```

### Tesseract Path

Uncomment and set the appropriate path for your system:

```python
# Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Linux/EC2
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
```

### Server Configuration

In the `if __name__ == "__main__":` section:
```python
app.run(host="0.0.0.0", port=5000, debug=False)
```

- `host="0.0.0.0"`: Accept connections from any IP (change to "127.0.0.1" for local only)
- `port=5000`: Server port (change as needed)
- `debug=False`: Disable debug mode in production

## Running the Server

### Development Mode

```bash
python app.py
```

Expected output:
```
 * Running on http://0.0.0.0:5000/
```

### Production Mode (Recommended)

Install Gunicorn:
```bash
pip install gunicorn
```

Run with Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

For Nginx reverse proxy setup, configure `/etc/nginx/sites-available/default` to forward requests to port 5000.

## API Endpoints

### 1. Health Check / Server Info

**GET** `/`

Returns available endpoints and descriptions.

**Response:**
```json
{
    "message": "YOLO + OCR + Background API",
    "routes": {
        "POST /detect": "Run YOLO detection",
        "POST /remove-bg": "Remove background",
        "POST /extract-text": "Extract text",
        "POST /find-all": "List all outputs for a user",
        "GET /outputs/<path:filename>": "Serve an output image"
    }
}
```

### 2. Health Status

**GET** `/health`

Check if the server is running.

**Response:**
```json
{
    "status": "success",
    "message": "Server healthy"
}
```

### 3. Object Detection (YOLO)

**POST** `/detect`

Detect objects in an image using YOLO model.

**Parameters (form-data):**
- `image` (file, required): Image file (JPG, JPEG, PNG)
- `_id` (string, required): User identifier for output organization

**Response:**
```json
{
    "status": "success",
    "output_url": "/outputs/user123/uuid_processed_detect.jpg",
    "detections": [
        {
            "class": "person",
            "confidence": 0.95,
            "bbox": [100, 50, 300, 400]
        }
    ],
    "object_types": ["person", "dog"]
}
```

**Error Response:**
```json
{
    "error": "Detection failed: [error details]"
}
```

### 4. Background Removal

**POST** `/remove-bg`

Remove background from an image.

**Parameters (form-data):**
- `image` (file, required): Image file (JPG, JPEG, PNG)
- `_id` (string, required): User identifier

**Response:**
```json
{
    "status": "success",
    "output_url": "/outputs/user123/uuid_processed_bg.jpg"
}
```

### 5. Text Extraction (OCR)

**POST** `/extract-text`

Extract text from an image using Tesseract OCR.

**Parameters (form-data):**
- `image` (file, required): Image file with text (JPG, JPEG, PNG)
- `_id` (string, required): User identifier

**Response:**
```json
{
    "status": "success",
    "extracted_text": "The extracted text from the image..."
}
```

### 6. Find All User Outputs

**POST** `/find-all`

List all processed outputs for a specific user.

**Parameters (form-data):**
- `_id` (string, required): User identifier

**Response:**
```json
{
    "images": [
        "/outputs/user123/uuid1_processed_detect.jpg",
        "/outputs/user123/uuid2_processed_bg.jpg"
    ]
}
```

### 7. Download Output Image

**GET** `/outputs/<user_id>/<filename>`

Download/view a processed image.

**Parameters:**
- `user_id`: User identifier folder
- `filename`: Image filename

**Response:** Binary image file

## Usage Examples

### Using cURL

**Detect Objects:**
```bash
curl -X POST http://localhost:5000/detect \
  -F "image=@photo.jpg" \
  -F "_id=user123"
```

**Remove Background:**
```bash
curl -X POST http://localhost:5000/remove-bg \
  -F "image=@photo.jpg" \
  -F "_id=user123"
```

**Extract Text:**
```bash
curl -X POST http://localhost:5000/extract-text \
  -F "image=@document.jpg" \
  -F "_id=user123"
```

**Get All Outputs:**
```bash
curl -X POST http://localhost:5000/find-all \
  -F "_id=user123"
```

### Using Python Requests

```python
import requests

BASE_URL = "http://localhost:5000"

# Object Detection
with open("photo.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/detect",
        files={"image": f},
        data={"_id": "user123"}
    )
    print(response.json())

# Background Removal
with open("photo.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/remove-bg",
        files={"image": f},
        data={"_id": "user123"}
    )
    print(response.json())

# Text Extraction
with open("document.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/extract-text",
        files={"image": f},
        data={"_id": "user123"}
    )
    print(response.json())

# List All Outputs
response = requests.post(
    f"{BASE_URL}/find-all",
    data={"_id": "user123"}
)
print(response.json())
```

### Using JavaScript/Fetch

```javascript
const userId = "user123";
const imageFile = document.getElementById("imageInput").files[0];

// Object Detection
const formData = new FormData();
formData.append("image", imageFile);
formData.append("_id", userId);

fetch("http://localhost:5000/detect", {
    method: "POST",
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Response Format

### Success Response

All successful responses include:
```json
{
    "status": "success",
    "output_url": "/outputs/user_id/filename.jpg",
    "additional_data": {}
}
```

### Error Response

```json
{
    "error": "Descriptive error message"
}
```

**Common Error Codes:**
- `400`: Bad request (missing parameters, invalid file type)
- `404`: File not found
- `500`: Server error during processing

## File Structure

```
yolo-api/
├── app.py                      # Main Flask application
├── object_identify.pt          # YOLO model (must exist)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── uploads/                    # Temporary upload storage (auto-created)
│   └── [temporary files]
└── outputs/                    # Persistent user outputs (auto-created)
    ├── user123/
    │   ├── uuid1_processed_detect.jpg
    │   └── uuid2_processed_bg.jpg
    └── user456/
        └── uuid3_processed_bg.jpg
```

## Troubleshooting

### Issue: "Model file not found"

**Solution:** Ensure `object_identify.pt` exists in the project root directory.

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').save('object_identify.pt')"
```

### Issue: "Tesseract not found" (OCR endpoint fails)

**Solution:** Install Tesseract for your OS and update the path in the code.

- **Windows:** Install from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux:** `sudo apt-get install tesseract-ocr`
- **Mac:** `brew install tesseract`

### Issue: "Invalid file type" or "No image provided"

**Solution:** Ensure you're uploading valid JPG/JPEG/PNG files with proper form-data encoding.

### Issue: Out of memory during detection

**Solution:** 
1. Use a smaller YOLO model: `yolov8n.pt` (nano) instead of `yolov8l.pt` (large)
2. Reduce image resolution before uploading
3. Run on a machine with more RAM or GPU support

### Issue: CORS errors in browser

**Solution:** Flask-CORS is enabled by default. If issues persist, ensure `CORS(app)` is called early in the application.

## Production Deployment

### Using Gunicorn + Nginx

1. **Install dependencies:**
```bash
pip install gunicorn
```

2. **Create systemd service** (`/etc/systemd/system/yolo-api.service`):
```ini
[Unit]
Description=YOLO API Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/home/ubuntu/yolo-api
ExecStart=/home/ubuntu/yolo-api/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

3. **Start service:**
```bash
sudo systemctl start yolo-api
sudo systemctl enable yolo-api
```

4. **Configure Nginx** (`/etc/nginx/sites-available/default`):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /outputs/ {
        alias /home/ubuntu/yolo-api/outputs/;
    }
}
```

### AWS EC2 Deployment

1. Launch EC2 instance (Ubuntu 20.04 or later)
2. Install Python 3.8+, Tesseract, and system dependencies
3. Clone project and follow installation steps
4. Use Gunicorn for production
5. Configure security groups to allow ports 80/443
6. Use SSL certificate with Let's Encrypt

### Docker Deployment

Create a `Dockerfile`:
```

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

```

## Performance Tips

- **Model Selection:** Use `yolov8n.pt` for speed, `yolov8l.pt` for accuracy
- **GPU Support:** Enable CUDA for 5-10x faster inference
- **Batch Processing:** Process multiple images sequentially; the server handles concurrent requests
- **Output Cleanup:** Use the `cleanup_old_outputs()` function for disk space management
- **Caching:** Consider implementing Redis for model caching in high-traffic scenarios

## License

Specify your license here (MIT, Apache 2.0, etc.)

## Support

For issues or questions, refer to:
- [YOLO Documentation](https://docs.ultralytics.com)
- [Rembg GitHub](https://github.com/danielgatis/rembg)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract)
- [Flask Documentation](https://flask.palletsprojects.com)