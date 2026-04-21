from fastapi import FastAPI, File

from model import ChestXRayModel
import torch
from torchvision.io import decode_image
import os
from datetime import datetime


app = FastAPI()

weights_path = "Lait-au-pole/chestxpert"
local_weights_path = "./model"
if os.path.exists(local_weights_path):
    weights_path = local_weights_path
model = ChestXRayModel().from_pretrained(weights_path)
labels = ["Atelectasis","Cardiomegaly","Consolidation",
          "Edema","Effusion","Emphysema",
          "Fibrosis","Hernia","Infiltration",
          "Mass","Nodule","Pleural_Thickening",
          "Pneumonia","Pneumothorax"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


@app.post("/predict")
async def predict(file: bytes = File()):

    # # Read the uploaded file
    # contents = await file.read()
    
    # # Save file for reference (optional)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"{timestamp}_{file.filename}"
    # os.makedirs("static/uploads", exist_ok=True)
    # file_path = os.path.join("static/uploads", filename)
    
    # with open(file_path, "wb") as f:
    #     f.write(contents)
    
    with torch.no_grad():
        image = torch.tensor(list(file),dtype=torch.uint8)
        image = decode_image(image, 'GRAY').to(device)
        pred = model.predict(image)
    predictions_dict = {label:prob for label, prob in zip(labels, pred)}
    return predictions_dict