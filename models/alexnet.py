import torch
import torch.nn as nn

"""
References:
Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. 
'Imagenet classification with deep convolutional neural networks. '
Communications of the ACM 60.6 (2017): 84-90.
"""

class AlexNet(nn.Module):
    def __init__(self, nclasses=1000, in_chans=3):
        super(AlexNet, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv2d(in_chans, 96, 11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        ])
        self.fc = nn.ModuleList([
            nn.Linear(5*5*256, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, nclasses)
        ])
    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = torch.reshape(x, (x.size(0), -1))
        for layer in self.fc:
            x = layer(x)
        return x

if __name__ == "__main__":
    T = torch.randn((16, 3, 224, 224))
    m = AlexNet()
    print(m)
    with torch.no_grad():
        y = m(T)
    print(y.shape)