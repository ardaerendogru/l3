import torch 
import torch.nn as nn

class ResnetHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ResnetHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)