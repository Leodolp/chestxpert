import torch.nn as nn
import torchvision
from torchvision.transforms import v2
import torch
from huggingface_hub import PyTorchModelHubMixin 

# EfficientNetB0 with 14 class output
class ChestXRayModel(nn.Module, PyTorchModelHubMixin,repo_url="chestxpert",
    license="mit",):
    def __init__(self, num_classes=14):
        super(ChestXRayModel, self).__init__()
        self.backbone = torchvision.models.efficientnet_b0(weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        orig_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(1, orig_conv.out_channels,
                        kernel_size=orig_conv.kernel_size,
                        stride=orig_conv.stride,
                        padding=orig_conv.padding,
                        bias=False)
        self.classifier = torch.nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        self.backbone.classifier = self.backbone.classifier[0]
        
        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(224),
            v2.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
    def predict(self, image: torch.Tensor) -> list: 
        self.eval()
        if len(image.shape) < 4:
            image = image.unsqueeze(0)
        image = self.transforms(image)
        logits = self.forward(image)
        pred = torch.sigmoid(logits).cpu().numpy()
        pred = pred.squeeze().tolist()
        return pred
