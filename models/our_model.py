from torch import nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

from models.face_recognition import MyFaceIdYOLOv8
from models.frcnn import MyFRCNN
from models.resnet9 import ResNet
from models.sphereface import sphere20a


class OurModel(nn.Module):

    def __init__(self, num_output_classes):
        super(OurModel, self).__init__()
        self.face_detection = MyFaceIdYOLOv8()
        self.m = ResNet()
        self.a = MyFRCNN(num_output_classes)
        self.d = sphere20a()
