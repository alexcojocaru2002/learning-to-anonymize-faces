import torch
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from models.face_recognition import MyFaceIdYOLOv8
from models.frcnn import MyFRCNN
from models.resnet9 import ResNet
from models.sphereface import sphere20a


class OurModel(nn.Module):

    def __init__(self, num_output_classes):
        super(OurModel, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.device = device

        self.face_detection = MyFaceIdYOLOv8()
        self.m = ResNet(device)

        backbone = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.IMAGENET1K_V1)
        self.a = FasterRCNN(backbone, num_classes=num_output_classes + 1)
        self.d = sphere20a(net_path='weights/sphere20a_20171020.pth')


        self.face_detection.model.to(device)
        self.m.to(device)
        self.a.to(device)
        self.d.to(device)