# YOLO V8 OBJECT DETECTION
The YOLO (You only look once )  which is famous  algorithm for the object detection. In this porject i used YOLOv8 pretrained model to detect the objects in the image and identify the categorty of the image which means the object class.
if the image had multiple objects  based on toatl confidence it will display the majority confidence category object and  display it in the JSON format.through simple web interface.
## Features
- Uses Ultralytics YOLOv8n pretrained model  
- Upload any image (supports multiple objects)  
- Detects all objects with bounding boxes and confidence  
- Returns dominant target class (highest total confidence)  
- Simple Flask web interface  
- Saves annotated results automatically  

## Tech Stack
- Python 
- Flask  
- YOLOv8 (Ultralytics)  
- OpenCV / Pillow  
- Torch

## Project Structure
YOLO_project
│
├── app.py                # Flask application
├── requirements.txt       # Dependencies
├── README.md              # Project description
│
├── templates/
│   └── index.html         # Web interface
│
└── static/
    |___images /           # images  
    ├── uploads/           # Uploaded images
    └── results/           # YOLO output images
## 2. create a virtual environment
python -m venv venv
venv\Scripts\activate
## 3.run the requirements.py file
pip install -r requirements.txt
## 4. run the app
python app.py

## API Endpoint
/detect (POST)

Accepts an image file and returns JSON with:

target: Dominant object class

detections: List of detected classes, confidence, and bounding boxes

it will display image with bounding box and with the calsses ,confindence and json format output below the image.

## Example
{
  "target": "person",
  "detections": [
    {"class_name": "person", "confidence": 0.91, "bbox": [120.4, 85.1, 240.6, 310.3]},
    {"class_name": "car", "confidence": 0.83, "bbox": [300.2, 150.4, 600.8, 400.7]}
  ]
}
