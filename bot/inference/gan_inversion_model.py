import torch.nn as nn

"""
Declaration of GAN Inversion modules
"""

class ResBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.act(self.conv(x))+self.act(self.conv2(self.act(self.conv1(x))))

class Encoder(nn.Module):
    def __init__(self, z_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
            ResBlock(64, 128),
            nn.AvgPool2d(2),
            ResBlock(128, 256),
            nn.AvgPool2d(2),
            ResBlock(256, 512),
            nn.AvgPool2d(2),
            ResBlock(512, 1024),
        )
        self.linear = nn.Linear(1024 * 4 * 4, 512)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.linear(x)
