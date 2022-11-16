import torch
import torch.nn as nn

"""
References:
He, Kaiming, et al. "Deep residual learning for image recognition." 
Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
"""

class ResBlock(nn.Module):
    # Residual building block
    # Seen in res18 and res34
    def __init__(self, in_chans, out_chans, stride=1):
        super(ResBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_chans, out_chans, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_chans, out_chans, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.relu2 = nn.ReLU()
        if stride!=1:
            self.projection = nn.Conv2d(in_chans, out_chans, 1, stride, padding=0)
    def forward(self, x):
        out = self.bn1(self.conv1(self.relu1(x)))
        out = self.bn2(self.conv2(out))
        # Perform 1x1 convolution if the dimension doesn't match
        if self.stride != 1:
            out += self.projection(x)
        else:
            out += x
        out = self.relu2(out)
        return out

class BottleneckBlock(nn.Module):
    # Bottleneck building block
    # Seen in res50/101/152
    def __init__(self, in_chans, out_chans, stride=1):
        super(BottleneckBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_chans, in_chans//2, 1, stride, padding=0)
        self.bn1 = nn.BatchNorm2d(in_chans//2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_chans//2, in_chans//2, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_chans//2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_chans//2, out_chans, 1, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_chans)
        self.relu3 = nn.ReLU()
        if stride==2:
            self.projection = nn.Conv2d(in_chans, out_chans, 1, stride, padding=0)
    def forward(self, x):
        out = self.bn1(self.conv1(self.relu1(x)))
        out = self.bn2(self.conv2(self.relu2(out)))
        out = self.bn3(self.conv3(out))
        # Perform 1x1 convolution if the dimension doesn't match
        if self.stride != 1:
            out += self.projection(x)        
        else:
            out += x
        out = self.relu2(out)
        return out

class ResidualNet(nn.Module):
    def __init__(self, nclass, in_chans, layers, bottleneck=False):
        super(ResidualNet, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv2d(in_chans, 64, 7, stride=2, padding=3), 
            nn.ReLU(), 
            nn.MaxPool2d(3, stride=2, padding=1)
            ])
        if not bottleneck:
            self.conv.extend([ResBlock(64, 64)] * layers[0])
            self.conv.extend([ResBlock(64, 128, 2)] + [ResBlock(128, 128)] * (layers[1]-1))
            self.conv.extend([ResBlock(128, 256, 2)] + [ResBlock(256, 256)] * (layers[2]-1))
            self.conv.extend([ResBlock(256, 512, 2)] + [ResBlock(512, 512)] * (layers[3]-1))
            self.fc = nn.Linear(7*7*512, nclass, bias=True)
        else:
            self.conv.extend([BottleneckBlock(64, 256)] + [BottleneckBlock(256, 256)]  * (layers[0]-1))
            self.conv.extend([BottleneckBlock(256, 512, 2)] + [BottleneckBlock(512, 512)] * (layers[1]-1))
            self.conv.extend([BottleneckBlock(512, 1024, 2)] + [BottleneckBlock(1024, 1024)] * (layers[2]-1))
            self.conv.extend([BottleneckBlock(1024, 2048, 2)] + [BottleneckBlock(2048, 2048)] * (layers[3]-1))       
            self.fc = nn.Linear(7*7*2048, nclass, bias=True)    
    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = torch.reshape(x, (x.size(0),-1))
        out = self.fc(x)
        return out

class ResNet18(ResidualNet):
    def __init__(self, nclass, in_chans):
        super(ResNet18, self).__init__(nclass, in_chans, [2, 2, 2, 2])
    
class ResNet34(ResidualNet):
    def __init__(self, nclass, in_chans):
        super(ResNet34, self).__init__(nclass, in_chans, [3, 4, 6, 3])

class ResNet50(ResidualNet):
    def __init__(self, nclass, in_chans):
        super(ResNet50, self).__init__(nclass, in_chans, [3, 4, 6, 3], bottleneck=True)

class ResNet101(ResidualNet):
    def __init__(self, nclass, in_chans):
        super(ResNet101, self).__init__(nclass, in_chans, [3, 4, 23, 3], bottleneck=True)

class ResNet152(ResidualNet):
    def __init__(self, nclass, in_chans):
        super(ResNet152, self).__init__(nclass, in_chans, [3, 8, 36, 3], bottleneck=True)

if __name__ == "__main__":
    T = torch.randn((10, 3, 224, 224))
    m = ResNet18(nclass=10, in_chans=3)
    print(m)
    with torch.no_grad():
        y = m(T)
    print(y.shape)