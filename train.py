import torch
import torch.nn as nn
from ultralytics import YOLO

class LightEnhancement(nn.Module):
    def _init_(self):
        super(LightEnhancement, self)._init_()
        self.enhance = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return torch.relu(self.enhance(x))

class AttentionModule(nn.Module):
    def _init_(self, in_channels):
        super(AttentionModule, self)._init_()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

import torch
import torch.nn as nn
from ultralytics import YOLO

class LightEnhancement(nn.Module):
    def __init__(self):
        super(LightEnhancement, self).__init__()
        self.enhance = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.enhance(x)

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class CustomYOLO(nn.Module):
    def __init__(self, model_path='yolov10n.pt', num_classes=1):
        super(CustomYOLO, self).__init__()
        self.yolo = YOLO(model_path)
        self.light_enhancement = LightEnhancement()
        self.attention = AttentionModule(in_channels=3)  # Input channels for attention

    def forward(self, x):
        x = self.light_enhancement(x)
        x = self.attention(x)
        x = self.yolo.model(x)  # Forward through YOLO model layers
        return x

    # Usage
model = CustomYOLO(model_path='yolov10n.pt', num_classes=1)


# Training with custom modules
model.yolo.train(data='base_file/data.yaml', epochs=50,imgsz=640)