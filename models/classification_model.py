import torch
import torch.nn as nn
from .backbones import CustomBackbone
from .backbones import resnet18, resnet34, resnet50, resnet101, resnet152
from .heads import ClassificationHead, ResnetHead

backbone_registry = {
    'custom_backbone': CustomBackbone,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152
}

head_registry = {
    'custom_linear_head': ClassificationHead,
    'resnet_linear_head': ResnetHead
}
class ClassificationModel(nn.Module):
    def __init__(self, backbone, head, out_features, num_classes):
        super(ClassificationModel, self).__init__()
        self.backbone = backbone_registry[backbone]()
        self.head = head_registry[head](out_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)