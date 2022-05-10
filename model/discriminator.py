import torch
import torch.nn as nn
import torch.nn.functional as F

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 5, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.conv1 = inconv(12, 64)
        self.conv2 = inconv(64, 128)
        self.conv3 = inconv(128, 256)
        self.conv4 = inconv(256, 256)
        self.conv5 = inconv(256, 512)
        

        self.classifier = nn.Sequential(
            nn.Linear(1024*1024*8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)

        print(x.shape)

        x = self.classifier(x)
        x = torch.sigmoid(x)

        return x

# def _test():
#     rand = torch.ones([4, 12, 256, 256]).cuda()
#     t = discriminator().cuda()

#     r = t(rand)
#     print(r)

# _test()