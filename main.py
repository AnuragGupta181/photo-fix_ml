#do not use server.py use this one

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from rembg import remove
from PIL import Image, ImageDraw, ImageFont
import os
import uuid
from flask_cors import CORS
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
ALLOWED = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_DIR'] = UPLOAD_DIR
app.config['OUTPUT_DIR'] = OUTPUT_DIR
app.config['ALLOWED'] = ALLOWED

os.makedirs(app.config['UPLOAD_DIR'], exist_ok=True)
os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)


MODEL_PATH = "best11.pt" 
model = YOLO(MODEL_PATH)


def allowed_file(name):
    return '.' in name and name.rsplit('.', 1)[1].lower() in app.config['ALLOWED']


def save_jpg(img: Image.Image, path: str):
    """Save a PIL Image as JPG. Convert if necessary."""
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        bg.save(path, "JPEG", quality=90)
    else:
        rgb = img.convert("RGB")
        rgb.save(path, "JPEG", quality=90)


def draw_boxes_and_save(image_path, boxes, classes, out_path):
    """
    Draw boxes and class labels on image and save as JPG.
    boxes: list of [x1, y1, x2, y2]
    classes: list of class names
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(x)) for x in box]
        draw.rectangle([x1, y1, x2, y2], width=2, outline="red")
        label = classes[i]
        if font:
            draw.text((x1 + 4, y1 + 4), label, fill="red", font=font)
        else:
            draw.text((x1 + 4, y1 + 4), label, fill="red")

    save_jpg(img, out_path)


def make_output_folder(user_id):
    path = os.path.join(app.config['OUTPUT_DIR'], str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


#  ROUTES
@app.route("/", methods=["GET"])
def info():
    return jsonify({
        "message": "YOLO + OCR + Background API",
        "routes": {
            "POST /detect (form-data: image, _id)": "Run YOLO detection",
            "POST /remove-bg (form-data: image, _id)": "Remove background",
            "POST /extract-text (form-data: image, _id)": "Extract text",
            "POST /find-all (form-data: _id)": "List all outputs for a user",
            "GET /outputs/<path:filename>": "Serve an output image"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "success", "message": "Server healthy"}), 200


#  DETECT
@app.route("/detect", methods=["POST"])
def detect():
    # expects form-data: image file and _id
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    if '_id' not in request.form:
        return jsonify({"error": "_id (user id) is required in form-data"}), 400

    file = request.files['image']
    user_id = request.form.get('_id')

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    fname = secure_filename(file.filename)
    uid = uuid.uuid4().hex
    upload_name = f"{uid}_{fname}"
    upload_path = os.path.join(app.config['UPLOAD_DIR'], upload_name)
    file.save(upload_path)

    try:
        results = model.predict(source=upload_path, save=False)
        res = results[0]

        boxes = []
        classes = []
        confidences = []

        for b in res.boxes:
            xy = b.xyxy[0].tolist()
            cls = int(b.cls[0]) if hasattr(b, "cls") else int(b.cls)
            conf = float(b.conf[0]) if hasattr(b, "conf") else float(b.conf)
            boxes.append(xy)
            classes.append(model.names[cls])
            confidences.append(conf)

        out_folder = make_output_folder(user_id)
        out_name = f"{uid}_processed_detect.jpg"
        out_path = os.path.join(out_folder, out_name)

        draw_boxes_and_save(upload_path, boxes, classes, out_path)

        try:
            os.remove(upload_path)
        except Exception:
            pass

        bboxes = [{"class": classes[i], "confidence": confidences[i], "bbox": [float(x) for x in boxes[i]]}
                  for i in range(len(boxes))]

        return jsonify({
            "status": "success",
            "output_url": f"/outputs/{user_id}/{out_name}",
            "detections": bboxes,
            "object_types": list(set(classes))
        })

    except Exception as e:
        try:
            os.remove(upload_path)
        except Exception:
            pass
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500


#  REMOVE BACKGROUND
@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    if '_id' not in request.form:
        return jsonify({"error": "_id (user id) is required in form-data"}), 400

    file = request.files['image']
    user_id = request.form.get('_id')

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    fname = secure_filename(file.filename)
    uid = uuid.uuid4().hex
    upload_name = f"{uid}_{fname}"
    upload_path = os.path.join(app.config['UPLOAD_DIR'], upload_name)
    file.save(upload_path)

    try:
        img = Image.open(upload_path)
        img_no_bg = remove(img)
        out_folder = make_output_folder(user_id)
        out_name = f"{uid}_processed_bg.jpg"
        out_path = os.path.join(out_folder, out_name)

        save_jpg(img_no_bg, out_path)

        try:
            os.remove(upload_path)
        except Exception:
            pass

        return jsonify({
            "status": "success",
            "output_url": f"/outputs/{user_id}/{out_name}"
        })

    except Exception as e:
        try:
            os.remove(upload_path)
        except Exception:
            pass
        return jsonify({"error": f"Background removal failed: {str(e)}"}), 500


#  EXTRACT TEXT
@app.route("/extract-text", methods=["POST"])
def extract_text():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    if '_id' not in request.form:
        return jsonify({"error": "_id (user id) is required in form-data"}), 400

    file = request.files['image']
    user_id = request.form.get('_id')

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    fname = secure_filename(file.filename)
    uid = uuid.uuid4().hex
    upload_name = f"{uid}_{fname}"
    upload_path = os.path.join(app.config['UPLOAD_DIR'], upload_name)
    file.save(upload_path)

    try:
        text = pytesseract.image_to_string(Image.open(upload_path))
        try:
            os.remove(upload_path)
        except Exception:
            pass

        return jsonify({
            "status": "success",
            "extracted_text": text.strip()
        })

    except Exception as e:
        try:
            os.remove(upload_path)
        except Exception:
            pass
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500


#  FIND ALL OUTPUTS FOR A USER
@app.route("/find-all", methods=["POST"])
def find_all():
    if '_id' not in request.form:
        return jsonify({"error": "_id (user id) is required in form-data"}), 400

    user_id = request.form.get('_id')
    folder = os.path.join(app.config['OUTPUT_DIR'], str(user_id))

    if not os.path.exists(folder):
        return jsonify({"images": []})

    files = sorted(os.listdir(folder))
    urls = [f"/outputs/{user_id}/{f}" for f in files]
    return jsonify({"images": urls})


#  SERVE OUTPUT FILES (static)
@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output(filename):
    safe_path = os.path.normpath(filename)
    if safe_path.startswith(".."):
        return jsonify({"error": "Invalid path"}), 400

    parts = safe_path.split(os.path.sep)
    if len(parts) < 2:
        return jsonify({"error": "Invalid output path"}), 400

    user_folder = parts[0]
    file_name = os.path.join(*parts[1:])  
    directory = os.path.join(app.config['OUTPUT_DIR'], user_folder)

    full_file = os.path.join(directory, file_name)
    if not os.path.isfile(full_file):
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(directory, file_name, as_attachment=False)


#  CLEANUP 
def cleanup_old_outputs(days=30):
    """Call this from a cron job or separate thread/process if needed."""
    cutoff = days * 24 * 3600
    now = os.path.getmtime
    base = app.config['OUTPUT_DIR']
    for root, _, files in os.walk(base):
        for f in files:
            p = os.path.join(root, f)
            if (os.path.getmtime(p) + cutoff) < os.path.getctime(p):
                try:
                    os.remove(p)
                except Exception:
                    pass



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
