from model import ChestXRayModel
from torchvision.io import decode_image
from torchvision.transforms.v2.functional import to_dtype
from torch import sigmoid
import argparse
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(image_path: str):
    weights_path = "Lait-au-pole/chestxpert"
    local_weights_path = "./model"
    if os.path.exists(local_weights_path):
        weights_path = local_weights_path
    with torch.no_grad():
        model = ChestXRayModel.from_pretrained(weights_path).to(device)
        image = decode_image(image_path, 'GRAY').to(device)
        pred = model.predict(image)
        return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser("python predict.py")
    parser.add_argument("--image", "-i", help="path to to the image for pathologie prediction", type=str)
    args = parser.parse_args()
    pred = main(args.image)
    labels = ["Atelectasis","Cardiomegaly","Consolidation",
          "Edema","Effusion","Emphysema",
          "Fibrosis","Hernia","Infiltration",
          "Mass","Nodule","Pleural_Thickening",
          "Pneumonia","Pneumothorax"]
    for i in range(len(labels)):
        print(f"{labels[i]}:\t {pred[i]:.3f}")
