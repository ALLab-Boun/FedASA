import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicConv2d, self).__init__()
        self.conv = conv3x3(in_planes, out_planes, stride)
        self.bn = nn.BatchNorm2d(out_planes)  # Optional, based on your has_bn flag
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)  # Remove this line if you don't want batch normalization
        x = self.leakyrelu(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, has_bn=True):
        super(SimpleCNN, self).__init__()
        self.layer1 = BasicConv2d(3, 16, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = BasicConv2d(16, 32, stride=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Adjust the size according to your input image size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
