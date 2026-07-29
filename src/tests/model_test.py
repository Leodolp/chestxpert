from src.model import ChestXRayModel 
import os
import torch
from torchvision.io import decode_image


def test_model():
    weights_path = "Lait-au-pole/chestxpert"
    local_weights_path = "./model"
    if os.path.exists(local_weights_path):
        weights_path = local_weights_path
    model = ChestXRayModel().from_pretrained(weights_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        test_sample = decode_image("data/00000001_000_Cardiomegalie.png", 'GRAY').to(device)
        model_prediction = model.predict(test_sample)
        correct_label_index = 1
        assert torch.argmax(torch.tensor(model_prediction)).cpu().numpy() == correct_label_index

if __name__ == "__main__":
    test_model()