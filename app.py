from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from PIL import Image
import io
import sys
import os

# Ajouter le dossier YOLOv5 au path
sys.path.append('./yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox

# Initialisation de l'application
app = FastAPI()

# Chargement du modèle YOLOv5
device = 'cpu'
model = DetectMultiBackend(weights='best.pt', device=device)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Lecture de l'image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = np.array(img)

        # Prétraitement avec YOLO
        img_resized = letterbox(img, new_shape=640)[0]
        img_resized = img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_resized = np.ascontiguousarray(img_resized)

        img_tensor = torch.from_numpy(img_resized).to(device)
        img_tensor = img_tensor.float()
        img_tensor /= 255.0  # Normalisation
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Inference
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        results = []
        for det in pred:
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    label = model.names[int(cls)]
                    results.append({
                        "label": label,
                        "confidence": float(conf),
                        "box": [int(x.item()) for x in xyxy]
                    })

        return JSONResponse(content={"detections": results})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
