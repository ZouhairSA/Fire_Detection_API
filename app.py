from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
import os
import sys

# Ajouter le chemin vers yolov5
YOLOV5_PATH = os.path.join(os.getcwd(), "yolov5")
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

from models.common import DetectMultiBackend
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

app = FastAPI()

# Charger le modèle
device = select_device("cpu")
model = DetectMultiBackend("best.pt", device=device)
model.eval()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img = np.array(image)

    # Prétraitement
    img_resized = letterbox(img, new_shape=640)[0]
    img_resized = img_resized.transpose((2, 0, 1))
    img_resized = np.ascontiguousarray(img_resized)

    img_tensor = torch.from_numpy(img_resized).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    detections = []
    for det in pred[0]:
        det = det.cpu().numpy()
        x1, y1, x2, y2, conf, cls = det
        detections.append({
            "class": int(cls),
            "confidence": round(float(conf), 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })

    return JSONResponse(content={"detections": detections})
