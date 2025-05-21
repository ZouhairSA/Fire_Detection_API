from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from PIL import Image
import io
from yolov5 import YOLOv5  # Assurez-vous que la lib yolov5 est bien installée (via pip install yolov5)

app = FastAPI()

# Charger le modèle YOLOv5
model = YOLOv5("yolo11n.pt", device="cpu")  # Chemin vers votre modèle

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    # Faire la prédiction
    results = model.predict(image_np)

    # Extraire les détections
    detections = results.pred[0]  # Détections du batch 0
    output = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        output.append({
            "class": int(cls),
            "confidence": round(conf, 3),
            "bbox": [round(x1), round(y1), round(x2), round(y2)]
        })

    return JSONResponse(content={"detections": output})
