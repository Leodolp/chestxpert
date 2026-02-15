import torch.nn as nn
import torchvision
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
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x