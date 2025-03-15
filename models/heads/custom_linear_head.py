import torch 
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ClassificationHead, self).__init__()
        self.flatten = nn.Flatten(2)
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.flatten(x).mean(-1)
        return self.fc(x)