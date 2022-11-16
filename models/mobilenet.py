import torch
import torch.nn as nn

"""
References:
Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." 
arXiv preprint arXiv:1704.04861 (2017).
"""

class MobileLayer(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1):
        super(MobileLayer, self).__init__()
        self.depthwise = nn.Conv2d(in_chans, in_chans, 3, stride, padding=1, groups=in_chans)
        self.pointwise = nn.Conv2d(in_chans, in_chans, 1, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_chans)
        self.relu1 = nn.ReLU()
        self.conv = nn.Conv2d(in_chans, out_chans, 1, 1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.relu2 = nn.ReLU()
    def forward(self, x):
        x = self.pointwise(self.depthwise(x))
        x = self.relu1(self.bn1(x))
        x = self.relu2(self.bn2(self.conv(x)))
        return x

class MobileNet(nn.Module):
    def __init__(self, nclass, in_chans):
        super(MobileNet, self).__init__()
        self.net = nn.ModuleList([
            nn.Conv2d(in_chans, 32, 3, stride=2, padding=1),
            MobileLayer(32, 64, 1),
            MobileLayer(64, 128, 2),
            MobileLayer(128, 128, 1),
            MobileLayer(128, 256, 2),
            MobileLayer(256, 256, 1),
            MobileLayer(256, 512, 2),
        ])
        self.net.extend([
            MobileLayer(512, 512, 1)
        ] * 5)
        self.net.extend([
            MobileLayer(512, 1024, 2),
            MobileLayer(1024, 1024, 1)
        ])
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, nclass)
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.reshape(x, (x.size(0), -1))
        x = self.fc(x)
        return x

if __name__ == "__main__":
    T = torch.randn((10, 3, 224, 224))
    m = MobileNet(2, 3)
    print(m)
    with torch.no_grad():
        y = m(T)
    print(y.shape)    

