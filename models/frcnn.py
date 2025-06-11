import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import resnet101, ResNet101_Weights, ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------- DALY Action Classes ----------
# Index 0 is background by convention in Faster R-CNN
action_classes = {
    1: "drinking",
    2: "phoning",
    3: "cleaning floor",
    4: "cleaning windows",
    5: "ironing",
    6: "folding textile",
    7: "playing harmonica",
    8: "taking photos",
    9: "vacuum cleaning",
    10: "watching TV"
}

class MyFRCNN(nn.Module):
    def __init__(self, num_action_classes):
        super().__init__()
        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")

        backbone = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = FasterRCNN(backbone, num_classes=num_action_classes + 1)

        # We will train this end to end but since the backbone resnet is pretrained on imagenet it should not be the longest training
        self.model.to(device)

    '''
    images: list of input images (tensors, shape [3, H, W])
    targets: list of dicts, each with:
        "boxes": Tensor[N, 4] (in [x1, y1, x2, y2] format)
        "labels": Tensor[N] (integer class labels)
        Optionally "image_id", "masks", etc
    '''
    def forward(self, images, targets=None):
        return self.model(images, targets)