import torch
import torch.nn as nn

class LeNet(nn.Module):
    """
    Reference:
    LeCun, Yann, et al. "Handwritten digit recognition with a back-propagation network." 
    Advances in neural information processing systems 2 (1989).
    """
    def __init__(self, nclass=10, in_chans=1):
        super(LeNet, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv2d(in_chans, 6, 5, 1, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(6, 16, 5, padding=0),
            nn.Sigmoid(),
            nn.MaxPool2d(2, stride=2)
        ])
        self.fc = nn.ModuleList([
            nn.Linear(5*5*16, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, nclass)
        ])
    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = torch.reshape(x, (x.size(0), -1))
        for layer in self.fc:
            x = layer(x)
        return x

if __name__ == "__main__":
    T = torch.randn((16, 1, 28, 28))
    m = LeNet()
    print(m)
    with torch.no_grad():
        y = m(T)
    print(y.shape)
        