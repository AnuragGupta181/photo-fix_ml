from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from rembg import remove
from PIL import Image
import os
import uuid
from flask_cors import CORS
import pytesseract

# Path for Tesseract (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# for lunix
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Create folders if not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load YOLO model
MODEL_PATH = "object_identify.pt"
model = YOLO(MODEL_PATH)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ----------------------------- YOLO DETECTION -----------------------------
@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(image_path)

    results = model.predict(source=image_path, save=True, project=app.config['OUTPUT_FOLDER'], name="detections", exist_ok=True)

    predictions = []
    for box in results[0].boxes:
        predictions.append({
            "class": model.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": [float(x) for x in box.xyxy[0]]
        })

    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], "detections")
    output_images = [f for f in os.listdir(output_dir) if unique_name.split('.')[0] in f]
    output_image = output_images[0] if output_images else None

    response = {
        "status": "success",
        "detections": predictions,
        "output_image_url": f"http://localhost:5000/outputs/{output_image}" if output_image else None
    }

    return jsonify(response)


# ----------------------------- OCR ROUTE -----------------------------
@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(image_path)

    try:
        text = pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500

    return jsonify({
        "status": "success",
        "extracted_text": text.strip()
    })


# ----------------------------- REMOVE BACKGROUND -----------------------------
@app.route('/remove-bg', methods=['POST'])
def remove_background():
    """Remove background from an uploaded image."""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_nobg_{filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(image_path)

    try:
        input_image = Image.open(image_path)
        output_image = remove(input_image)  # using rembg
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], unique_name)

        # 🩹 Fix: Convert RGBA → RGB if saving as JPG
        ext = os.path.splitext(unique_name)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            # Create white background for JPG
            bg = Image.new("RGB", output_image.size, (255, 255, 255))
            bg.paste(output_image, mask=output_image.split()[3])  # use alpha channel as mask
            bg.save(output_path, "JPEG")
        else:
            output_image.save(output_path, "PNG")

    except Exception as e:
        return jsonify({"error": f"Background removal failed: {str(e)}"}), 500

    return jsonify({
        "status": "success",
        "output_image_url": f"http://localhost:5000/outputs/{unique_name}"
    })


# ----------------------------- HEALTH & HOME -----------------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "YOLO Detection + OCR + Background API is running 🚀",
        "endpoints": {
            "POST /detect": "Upload image for YOLO object detection",
            "POST /extract-text": "Extract text from image using OCR",
            "POST /remove-bg": "Remove image background",
            "POST /add-bg": "Add custom background color or image"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "success", "message": "Server healthy"}), 200


@app.route('/outputs/<path:filename>')
def get_output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
