from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
from collections import defaultdict

app = Flask(__name__)

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Create folders
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    detections = []
    json_output = {}

    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("index.html", error="No image uploaded!")

        file = request.files["file"]
        upload_path = os.path.join("static/uploads", file.filename)
        file.save(upload_path)

        # Run YOLO detection
        results = model(upload_path)
        output_path = os.path.join("static/results", f"result_{file.filename}")
        results[0].save(filename=output_path)

        # Extract detections
        class_conf = defaultdict(float)
        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_name": cls_name,
                "confidence": round(conf, 2),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })

            # Add confidence to class total
            class_conf[cls_name] += conf

        # Find class with highest total confidence (dominant target)
        target_class = max(class_conf, key=class_conf.get) if class_conf else None

        json_output = {
            "target": target_class,
            "detections": detections
        }

        result_image = output_path

    return render_template("index.html",
                           result_image=result_image,
                           detections=detections,
                           json_output=json_output)


@app.route("/detect", methods=["POST"])
def detect_api():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    upload_path = os.path.join("static/uploads", file.filename)
    file.save(upload_path)

    # Run YOLO detection
    results = model(upload_path)

    detections = []
    class_conf = defaultdict(float)

    for box in results[0].boxes:
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_name": cls_name,
            "confidence": round(conf, 2),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        })
        class_conf[cls_name] += conf

    target_class = max(class_conf, key=class_conf.get) if class_conf else None

    return jsonify({
        "target": target_class,
        "detections": detections
    })


if __name__ == "__main__":
    app.run(debug=True)
