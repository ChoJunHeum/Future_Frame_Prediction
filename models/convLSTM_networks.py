import torch
import torch.nn as nn

from models.spectral_normalization import SpectralNorm
from models.convLSTM import ConvLSTM

# 입력으로 4장의 이미지
# conv -> Spectral normalization -> FReLU
# conv -> Spectral normalization -> BN -> FReLU * 4
# convLSTM * 3
# Upconv -> Spectral normalization -> BN -> FReLU * 4
# Upconv -> Spectral normalization



class FReLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.depthwise_conv_bn = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel))

    def forward(self, x):
        funnel_x = self.depthwise_conv_bn(x)
        return torch.max(x, funnel_x)



class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.sconv = nn.Sequential(
            SpectralNorm(self.conv),
            FReLU(out_ch)
        )

    def forward(self, x):
        out = self.sconv(x)
        return out


class downCell(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.sconv = nn.Sequential(
            nn.MaxPool2d(2),
            SpectralNorm(self.conv),
            nn.BatchNorm2d(out_ch),
            FReLU(out_ch)
        )

    def forward(self, x):
        out = self.sconv(x)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = downCell(3, 64)
        self.conv2 = downCell(64, 128)
        self.conv3 = downCell(128, 256)
        self.conv4 = downCell(256, 512)

    def forward(self, x):
        latents = []

        for i in range(4):
            frame = x[:,i*3:(i+1)*3, :, :]
            out = self.conv1(frame)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)

            latents.append(out)

        return torch.stack(latents)


class upCell(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.sconv = nn.Sequential(
            SpectralNorm(self.conv),
            nn.BatchNorm2d(out_ch),
            FReLU(out_ch)
        )

    def forward(self, x):
        out = self.sconv(x)
        return out


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.sconv = nn.Sequential(
            SpectralNorm(self.conv)
        )
    
    def forward(self, x):
        out = self.sconv(x)
        return out



class Decoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = upCell(in_ch, 256)
        self.conv2 = upCell(256, 128)
        self.conv3 = upCell(128, 64)
        self.conv4 = outconv(64, 3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out


class ConvLstmGenerator(nn.Module):
    def __init__(self):
        super(ConvLstmGenerator, self).__init__()
        self.encoder = Encoder()
        # self.convlstm = nn.Sequential(
        #     ConvLSTM(512, [128, 128, 128, 128], (5,5), 4, True, True, False),
        #     ConvLSTM(512, [128, 128, 128, 128], (5,5), 4, True, True, False),
        #     ConvLSTM(512, [128, 128, 128, 128], (5,5), 4, True, True, False)
        # )
        self.convlstm = ConvLSTM(512, [128, 128, 128], (5,5), 3, False, True, False)

        self.decoder = Decoder(128)

    def forward(self, x):
        # print("raw: ",x.shape)
        out = self.encoder(x)
        # print("encoder: ",out.shape)

        out, _ = self.convlstm(out)
        # print("convlstm: ",out[0].shape)

        out = out[0][:,-1,:,:,:]
        # print("convlstm: ",out.shape)

        out = self.decoder(out)
        # print("decoder: ",out.shape)
        # quit()
        return out

    
if __name__ == "__main__":
    rand = torch.ones([4, 12, 256, 256]).cuda()
    t = Generator(12, 3).cuda()

    r = t(rand)
    print(r.shape)
    print(r.grad_fn)
    print(r.requires_grad)
    print(r)