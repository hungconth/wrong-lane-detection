import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import glob
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from torch import nn
import torchvision 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms

classes = ["green", "red", "yellow"]
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bacNor = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.bacNor1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.bacNor2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.bacNor3 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.bacNor1(self.pool(F.relu(self.conv1(x))))
        x = self.bacNor2(self.pool(F.relu(self.conv2(x))))
        x = self.bacNor3(self.pool(F.relu(self.conv3(x))))
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

from numpy.core.fromnumeric import resize
test_transforms = transforms.Compose([
        transforms.ToPILImage(),                              
        transforms.Resize((30,80)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def Predicted(img):
    device = torch.device('cpu')
    model = Net()
    model.load_state_dict(torch.load("class_color_traffic_light.pth", map_location=device))
    img = test_transforms(img)
    img = img[None, :]
    imge = model(img)
    _, predicted = torch.max(imge, dim=1)
    return classes[predicted]
img = cv2.imread(r"./img/IMG_0266.JPG")
print(Predicted(img))