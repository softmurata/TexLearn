import torch.nn as nn
import torch.nn.functional as F
import torch


# Unet
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(num_features=in_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):

        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn1(self.conv1(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn2(self.conv2(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn3(self.conv3(x)))

        return x

class UnetGenerator(nn.Module):
    # model structure Unet
    def __init__(self, in_channels, out_channels, mode='nearest'):
        super(UnetGenerator, self).__init__()
        self.mode = mode
        # not use max pooling
        # input => 3 * 512 * 512
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1)  # 64 * 256 * 256
        self.bn1 = nn.BatchNorm2d(num_features=64)  # 64 * 256 * 256
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)  # 128 * 128 * 128
        self.bn2 = nn.BatchNorm2d(num_features=128)  # 128 * 256 * 256
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)  # 256 * 64 * 64
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)  # 512 * 32 * 32
        self.bn4 = nn.BatchNorm2d(num_features=512)  # 512 * 32 * 32
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)  # 1024 * 16 * 16
        self.bn5 = nn.BatchNorm2d(num_features=1024)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)  # 1024 * 8 * 8
        self.bn6 = nn.BatchNorm2d(num_features=1024)

        # Decoder side
        if self.mode:
            self.upsample1 = nn.Upsample(scale_factor=2, mode=self.mode)  # 1024 * 16 * 16(?)
            self.upconv1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, stride=1, padding=1)  # 512 * 16 * 16
            self.upbn1 = nn.BatchNorm2d(num_features=512)
            self.upsample2 = nn.Upsample(scale_factor=2, mode=self.mode)  # 512 * 32 * 32
            self.upconv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)  # 256 * 32 * 32
            self.upbn2 = nn.BatchNorm2d(num_features=256)
            self.upsample3 = nn.Upsample(scale_factor=2, mode=self.mode)  # 256 * 64 * 64
            self.upconv3 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)  # 128 * 64 * 64
            self.upbn3 = nn.BatchNorm2d(num_features=128)
            self.upsample4 = nn.Upsample(scale_factor=2, mode=self.mode)  # 128 * 128 * 128
            self.upconv4 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)  # 64 * 128 * 128
            self.upbn4 = nn.BatchNorm2d(num_features=64)
            self.upsample5 = nn.Upsample(scale_factor=2, mode=self.mode)  # 64 * 256 * 256
            self.upconv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)  # 32 * 256 * 256
            self.upbn5 = nn.BatchNorm2d(num_features=32)
            self.upsample6 = nn.Upsample(scale_factor=2, mode=self.mode)  # 32 * 512 * 512
            self.final_conv = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1)  # 3 * 512 * 512

        else:
            self.upconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, padding=0)  # 1024 * 16 * 16
            self.convblock1 = ConvBlock(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1)  # 1024 * 16 * 16
            self.upconv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)  # 512 * 32 * 32
            self.convblock2 = ConvBlock(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)  # 512 * 32 * 32
            self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)  # 256 * 64 * 64
            self.convblock3 = ConvBlock(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)  # 256 * 64 * 64
            self.upconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)  # 128 * 128 * 128
            self.convblock4 = ConvBlock(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)  # 128 * 128 * 128
            self.upconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)  # 64 * 256 * 256
            self.convblock5 = ConvBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)  # 64 * 256 * 256
            self.final_conv = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=2, stride=2, padding=0)  # 3 * 512 * 512

    def forward(self, x):
        # input => (batch_size, 3, 512, 512)
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn1(self.conv1(x)))  # (batch_size, 64, 256, 256)
        x1 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn2(self.conv2(x)))  # (batch_size, 128, 128, 128)
        x2 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn3(self.conv3(x)))  # (batch_size, 256, 64, 64)
        x3 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn4(self.conv4(x)))  # (batch_size, 512, 32, 32)
        x4 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn5(self.conv5(x)))  # (batch_size, 1024, 16, 16)
        x5 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn6(self.conv6(x)))  # (batch_size, 1024, 8, 8)

        # torch.cat([xi, x], dim=1) => skip connection:  tensor shape = (batch_size, channels, height, width)

        if self.mode:
            x = self.upsample1(x)  # (batch_size, 1024, 16, 16)
            x = torch.cat([x5, x], dim=1)  # (batch_size, 2048, 16, 16)
            x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(
                self.upbn1(self.upconv1(x)))  # (batch_size, 512, 16, 16)

            x = self.upsample2(x)  # (batch_size, 512, 32, 32)
            x = torch.cat([x4, x], dim=1)  # (batch_size, 1024, 32, 32)
            x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(
                self.upbn2(self.upconv2(x)))  # (batch_size, 256, 32, 32)

            x = self.upsample3(x)  # (batch_size, 256, 64, 64)
            x = torch.cat([x3, x], dim=1)  # (batch_size, 512, 64, 64)
            x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(
                self.upbn3(self.upconv3(x)))  # (batch_size, 128, 64, 64)

            x = self.upsample4(x)  # (batch_size, 128, 128, 128)
            x = torch.cat([x2, x], dim=1)  # (batch_size, 256, 128, 128)
            x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(
                self.upbn4(self.upconv4(x)))  # (batch_size, 64, 128, 128)

            x = self.upsample5(x)  # (batch_size, 64, 256, 256)
            x = torch.cat([x1, x], dim=1)  # (batch_size, 128, 256, 256)
            x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(
                self.upbn5(self.upconv5(x)))  # (batch_size, 32, 256, 256)

            x = nn.Tanh()(
                self.final_conv(self.upsample6(x)))  # (batch_size, 3, 512, 512)

        else:
            x = self.upconv1(x)  # (batch_size, 1024, 16, 16)
            x = torch.cat([x5, x], dim=1)  # (batch_size, 2048, 16, 16)
            x = self.convblock1(x)  # (batch_size, 1024, 16, 16)

            x = self.upconv2(x)  # (batch_size, 512, 32, 32)
            x = torch.cat([x4, x], dim=1)  # (batch_size, 1024, 32, 32)
            x = self.convblock2(x)  # (batch_size, 512, 32, 32)

            x = self.upconv3(x)  # (batch_size, 256, 64, 64)
            x = torch.cat([x3, x], dim=1)  # (batch_size, 512, 64, 64)
            x = self.convblock3(x)  # (batch_size, 256, 64, 64)

            x = self.upconv4(x)  # (batch_size, 128, 128, 128)
            x = torch.cat([x2, x], dim=1)  # (batch_size, 256, 128, 128)
            x = self.convblock4(x)  # (batch_size, 128, 128, 128)

            x = self.upconv5(x)  # (batch_size, 64, 256, 256)
            x = torch.cat([x1, x], dim=1)  # (batch_size, 128, 256, 256)
            x = self.convblock5(x)  # (batch_size, 64, 256, 256)

            x = nn.Tanh()(self.final_conv(x))  # (batch_size, 3, 512, 512)

        print('output shape:', x.shape)

        return x


# ResNet
class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResConv, self).__init__()
        # case => input image size => (512, 512)
        middle_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=middle_channels)
        self.conv2 = nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=middle_channels)
        self.conv3 = nn.Conv2d(in_channels=middle_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)

        self.skip_connect_conv = nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        x1 = x
        x = nn.ReLU(inplace=True)(self.bn2(self.conv2(x)))
        x = nn.ReLU(inplace=True)(self.bn3(self.conv3(x)))
        if x1.shape[1] != x.shape[1]:
            x1 = self.skip_connect_conv(x1)
        x = x1 + x  # skip connection(add operation)
        x = nn.ReLU(inplace=True)(x)

        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers):
        super(ResBlock, self).__init__()

        self.module_list = [
            ResConv(in_channels=in_channels, out_channels=out_channels)
        ]
        self.module_list += [
            ResConv(in_channels=out_channels, out_channels=out_channels) for _ in range(num_layers-1)
        ]

    def forward(self, x):
        for block in self.module_list:
            x = block(x)

        return x


# average pooling
class GlobalAveragePooling2d(nn.Module):
    def __init__(self, device='cpu'):
        super(GlobalAveragePooling2d, self).__init__()

    def forward(self, x):

        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))


class ResNet18(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet18, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2,
                               padding=3)  # 64 * 256 * 256
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 * 128 * 128

        self.resblock1 = ResBlock(in_channels=64, out_channels=64, num_layers=2)  # (batch_size, 64, 128, 128)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=2, padding=1)  # (batch_size, 128, 64, 64)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.resblock2 = ResBlock(in_channels=128, out_channels=128, num_layers=2)  # (batch_size, 128, 64, 64)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=2, padding=1)  # (batch_size, 256, 32, 32)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.resblock3 = ResBlock(in_channels=256, out_channels=256, num_layers=2)  # (batch_size, 256, 32, 32)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=3, stride=2, padding=1)  # (batch_size, 512, 16, 16)
        self.bn4 = nn.BatchNorm2d(num_features=512)
        self.resblock4 = ResBlock(in_channels=512, out_channels=512, num_layers=2)  # (batch_size, 512, 16, 16)

        # decoder
        self.upconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)  # (batch_size, 256, 32, 32)
        self.upbn1 = nn.BatchNorm2d(num_features=256)
        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)  # (batch_size, 128, 64, 64)
        self.upbn2 = nn.BatchNorm2d(num_features=128)
        self.upconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)  # (batch_size, 64, 128, 128)
        self.upbn3 = nn.BatchNorm2d(num_features=64)
        self.upconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)  # (batch_size, 64, 256, 256)
        self.upbn4 = nn.BatchNorm2d(num_features=64)
        self.upconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=2, stride=2, padding=0)  # (batch_size, 3, 512, 512)

    def forward(self, x):
        # Encoder
        # first layer
        x = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))  # (batch_size, 64, 256, 256)
        x = self.maxpool1(x)  # (batch_size, 64, 128, 128)

        # resblock
        x = self.resblock1(x)  # (batch_size, 64, 128, 128)
        x = nn.ReLU(inplace=True)(self.bn2(self.conv2(x)))  # (batch_size, 128, 64, 64)

        x = self.resblock2(x)  # (batch_size, 128, 64, 64)
        x = nn.ReLU(inplace=True)(self.bn3(self.conv3(x)))  # (batch_size, 256, 32, 32)

        x = self.resblock3(x)  # (batch_size, 256, 32, 32)
        x = nn.ReLU(inplace=True)(self.bn4(self.conv4(x)))  # (batch_size, 512, 16, 16)

        x = self.resblock4(x)  # (batch_size, 512, 16, 16)

        # Decoder
        x = nn.ReLU(inplace=True)(self.upbn1(self.upconv1(x)))  # (batch_size, 256, 32, 32)
        x = nn.ReLU(inplace=True)(self.upbn2(self.upconv2(x)))  # (batch_size, 128, 64, 64)
        x = nn.ReLU(inplace=True)(self.upbn3(self.upconv3(x)))  # (batch_size, 64, 128, 128)
        x = nn.ReLU(inplace=True)(self.upbn4(self.upconv4(x)))  # (batch_size, 64, 256, 256)
        x = nn.ReLU(inplace=True)(self.upconv5(x))  # (batch_size, 3, 512, 512)

        print('generator output shape:', x.shape)

        return x


class ResNet34(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResNet34, self).__init__()
        # Same structure
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2,
                               padding=3)  # 64 * 256 * 256
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 * 128 * 128

        # encoder
        self.resblock1 = ResBlock(in_channels=64, out_channels=64, num_layers=3)  # 64 * 128 * 128
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)  # 128 * 64 * 64
        self.bn2 = nn.BatchNorm2d(num_features=128)

        self.resblock2 = ResBlock(in_channels=128, out_channels=128, num_layers=4)  # 128 * 64 * 64
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)  # 256 * 32 * 32
        self.bn3 = nn.BatchNorm2d(num_features=256)

        self.resblock3 = ResBlock(in_channels=256, out_channels=256, num_layers=6)  # 256 * 32 * 32
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)  # 512 * 16 * 16
        self.bn4 = nn.BatchNorm2d(num_features=512)

        self.resblock4 = ResBlock(in_channels=512, out_channels=512, num_layers=3)  # 512 * 16 * 16

        # decoder
        self.upconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)  # 256 * 32 * 32
        self.upbn1 = nn.BatchNorm2d(num_features=256)
        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)  # 128 * 64 * 64
        self.upbn2 = nn.BatchNorm2d(num_features=128)
        self.upconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)  # 64 * 128 * 128
        self.upbn3 = nn.BatchNorm2d(num_features=64)
        self.upconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)  # 64 * 256 * 256
        self.upbn4 = nn.BatchNorm2d(num_features=64)
        self.upconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=2, stride=2, padding=0)  # 3 * 512 * 512

    def forward(self, x):

        # first layer
        x = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))  # (batch_size, 64, 256, 256)
        x = self.maxpool1(x)  # (batch_size, 64, 128, 128)

        # encoder
        x = self.resblock1(x)  # (batch_size, 64, 128, 128)
        x = nn.ReLU(inplace=True)(self.bn2(self.conv2(x)))  # (batch_size, 128, 64, 64)
        x = self.resblock2(x)  # (batch_size, 128, 64, 64)
        x = nn.ReLU(inplace=True)(self.bn3(self.conv3(x)))  # (batch_size, 256, 32, 32)
        x = self.resblock3(x)  # (batch_size, 256, 32, 32)
        x = nn.ReLU(inplace=True)(self.bn4(self.conv4(x)))  # (batch_size, 512, 16, 16)
        x = self.resblock4(x)  # (batch_size, 512, 16, 16)

        # decoder
        x = nn.ReLU(inplace=True)(self.upbn1(self.upconv1(x)))  # (batch_size, 256, 32, 32)
        x = nn.ReLU(inplace=True)(self.upbn2(self.upconv2(x)))  # (batch_size, 128, 64, 64)
        x = nn.ReLU(inplace=True)(self.upbn3(self.upconv3(x)))  # (batch_size, 64, 128, 128)
        x = nn.ReLU(inplace=True)(self.upbn4(self.upconv4(x)))  # (batch_size, 64, 256, 256)
        x = nn.ReLU(inplace=True)(self.upconv5(x))  # (batch_size, 3, 512, 512)
        print('generator output shape:', x.shape)

        return x



class Resnet50(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resnet50, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2,
                               padding=3)  # 64 * 256 * 256
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 * 128 * 128

        # encoder
        self.resblock1 = ResBlock(in_channels=64, out_channels=256, num_layers=3)  # 256 * 128 * 128
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2,
                               padding=1)  # 128 * 64 * 64
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.resblock2 = ResBlock(in_channels=128, out_channels=512, num_layers=4)  # 512 * 64 * 64
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=2,
                               padding=1)  # 256 * 32 * 32
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.resblock3 = ResBlock(in_channels=256, out_channels=1024, num_layers=6)  # 1024 * 32 * 32
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2,
                               padding=1)  # 512 * 16 * 16
        self.bn4 = nn.BatchNorm2d(num_features=512)
        self.resblock4 = ResBlock(in_channels=512, out_channels=2048, num_layers=3)  # 2048 * 16 * 16

        # decoder
        self.upconv1 = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=2, stride=2,
                                          padding=0)  # 1024 * 32 * 32
        self.upbn1 = nn.BatchNorm2d(num_features=1024)
        self.upconv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2,
                                          padding=0)  # 512 * 64 * 64
        self.upbn2 = nn.BatchNorm2d(num_features=512)
        self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2,
                                          padding=0)  # 256 * 128 * 128
        self.upbn3 = nn.BatchNorm2d(num_features=256)
        self.upconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2,
                                          padding=0)  # 128 * 256 * 256
        self.upbn4 = nn.BatchNorm2d(num_features=128)
        self.upconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2,
                                          padding=0)  # 64 * 512 * 512
        self.upbn5 = nn.BatchNorm2d(num_features=64)
        self.upconv6 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1,
                                          padding=0)  # 3 * 512 * 512

        # not need, if you want to use pretrained resnet weight model,
        # you have to delete pooling and fully-connected layer
        # self.avgpool = GlobalAveragePooling2d()
        # self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        # self.fc2 = nn.Linear(in_features=1024, out_features=out_channels)

    def forward(self, x):
        # first layer
        x = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))  # (batch_size, 64, 256, 256)
        # max pooling
        x = self.maxpool1(x)  # (batch_size, 64, 128, 128)

        # resblock1
        x = self.resblock1(x)  # (batch_size, 256, 128, 128)
        x = nn.ReLU(inplace=True)(self.bn2(self.conv2(x)))  # (batch_size, 512, 64, 64)
        # resblock2
        x = self.resblock2(x)  # (batch_size, 512, 64, 64)
        x = nn.ReLU(inplace=True)(self.bn3(self.conv3(x)))  # (batch_size, 1024, 32, 32)
        # resblock3
        x = self.resblock3(x)  # (batch_size, 1024, 32, 32)
        x = nn.ReLU(inplace=True)(self.bn4(self.conv4(x)))  # (batch_size, 2048, 16, 16)
        # resblock4
        x = self.resblock4(x)  # (batch_size, 2048, 16, 16)

        # decoder
        x = nn.ReLU(inplace=True)(self.upbn1(self.upconv1(x)))  # (batch_size, 1024, 32, 32)
        x = nn.ReLU(inplace=True)(self.upbn2(self.upconv2(x)))  # (batch_size, 512, 64, 64)
        x = nn.ReLU(inplace=True)(self.upbn3(self.upconv3(x)))  # (batch_size, 256, 128, 128)
        x = nn.ReLU(inplace=True)(self.upbn4(self.upconv4(x)))  # (batch_size, 128, 256, 256)
        x = nn.ReLU(inplace=True)(self.upbn5(self.upconv5(x)))  # (batch_size, 64, 512, 512)
        x = self.upconv6(x)  # (batch_size, 3, 512, 512)

        print('resnet generator shape:', x.shape)

        return x


class ResNet101(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet101, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2,
                               padding=3)  # 64 * 256 * 256
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 * 128 * 128

        # encoder
        self.resblock1 = ResBlock(in_channels=64, out_channels=256, num_layers=3)  # 256 * 128 * 128
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)  # 128 * 64 * 64
        self.bn2 = nn.BatchNorm2d(num_features=128)

        self.resblock2 = ResBlock(in_channels=128, out_channels=512, num_layers=4)  # 512 * 64 * 64
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1)  # 256 * 32 * 32
        self.bn3 = nn.BatchNorm2d(num_features=256)

        self.resblock3 = ResBlock(in_channels=256, out_channels=1024, num_layers=23)  # 1024 * 32 * 32
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1)  # 512 * 16 * 16
        self.bn4 = nn.BatchNorm2d(num_features=512)

        self.resblock4 = ResBlock(in_channels=512, out_channels=2048, num_layers=3)  # 2048 * 16 * 16

        # decoder
        self.upconv1 = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=2, stride=2, padding=0)  # 1024 * 32 * 32
        self.upbn1 = nn.BatchNorm2d(num_features=1024)
        self.upconv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2,
                                          padding=0)  # 512 * 64 * 64
        self.upbn2 = nn.BatchNorm2d(num_features=512)
        self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2,
                                          padding=0)  # 256 * 128 * 128
        self.upbn3 = nn.BatchNorm2d(num_features=256)
        self.upconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2,
                                          padding=0)  # 128 * 256 * 256
        self.upbn4 = nn.BatchNorm2d(num_features=128)
        self.upconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2,
                                          padding=0)  # 64 * 512 * 512
        self.upbn5 = nn.BatchNorm2d(num_features=64)
        self.final_conv = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0)  # 3 * 512 * 512

    def forward(self, x):
        # first layer
        x = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))  # (batch_size, 64, 256, 256)
        x = self.maxpool1(x)  # (batch_size, 64, 128, 128)

        # resblock
        x = self.resblock1(x)  # (batch_size, 256, 128, 128)
        x = nn.ReLU(inplace=True)(self.bn2(self.conv2(x)))  # (batch_size, 128, 64, 64)
        x = self.resblock2(x)  # (batch_size, 512, 64, 64)
        x = nn.ReLU(inplace=True)(self.bn3(self.conv3(x)))  # (batch_size, 256, 32, 32)
        x = self.resblock3(x)  # (batch_size, 1024, 32, 32)
        x = nn.ReLU(inplace=True)(self.bn4(self.conv4(x)))  # (batch_size, 512, 16, 16)
        x = self.resblock4(x)  # (batch_size, 2048, 16, 16)

        # decoder
        x = nn.ReLU(inplace=True)(self.upbn1(self.upconv1(x)))  # (batch_size, 1024, 32, 32)
        x = nn.ReLU(inplace=True)(self.upbn2(self.upconv2(x)))  # (batch_size, 512, 64, 64)
        x = nn.ReLU(inplace=True)(self.upbn3(self.upconv3(x)))  # (batch_size, 256, 128, 128)
        x = nn.ReLU(inplace=True)(self.upbn4(self.upconv4(x)))  # (batch_size, 128, 256, 256)
        x = nn.ReLU(inplace=True)(self.upbn5(self.upconv5(x)))  # (batch_size, 64, 512, 512)
        x = self.final_conv(x)  # (batch_size, 3, 512, 512)
        print('generator output shape:', x.shape)

        return x


class ResNet152(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet152, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2,
                               padding=3)  # 64 * 256 * 256
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64 * 128 * 128

        # encoder
        self.resblock1 = ResBlock(in_channels=64, out_channels=256, num_layers=3)  # 256 * 128 * 128
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)  # 128 * 64 * 64
        self.bn2 = nn.BatchNorm2d(num_features=128)

        self.resblock2 = ResBlock(in_channels=128, out_channels=512, num_layers=8)  # 512 * 64 * 64
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1)  # 256 * 32 * 32
        self.bn3 = nn.BatchNorm2d(num_features=256)

        self.resblock3 = ResBlock(in_channels=256, out_channels=1024, num_layers=36)  # 1024 * 32 * 32
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1)  # 512 * 16 * 16
        self.bn4 = nn.BatchNorm2d(num_features=512)

        self.resblock4 = ResBlock(in_channels=512, out_channels=2048, num_layers=3)  # 2048 * 16 * 16

        # decoder
        self.upconv1 = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=2, stride=2,
                                          padding=0)  # 1024 * 32 * 32
        self.upbn1 = nn.BatchNorm2d(num_features=1024)
        self.upconv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2,
                                          padding=0)  # 512 * 64 * 64
        self.upbn2 = nn.BatchNorm2d(num_features=512)
        self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2,
                                          padding=0)  # 256 * 128 * 128
        self.upbn3 = nn.BatchNorm2d(num_features=256)
        self.upconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2,
                                          padding=0)  # 128 * 256 * 256
        self.upbn4 = nn.BatchNorm2d(num_features=128)
        self.upconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2,
                                          padding=0)  # 64 * 512 * 512
        self.upbn5 = nn.BatchNorm2d(num_features=64)
        self.final_conv = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1,
                                             padding=0)  # 3 * 512 * 512

    def forward(self, x):
        # first layer
        x = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))  # (batch_size, 64, 256, 256)
        x = self.maxpool1(x)  # (batch_size, 64, 128, 128)

        # resblock
        x = self.resblock1(x)  # (batch_size, 256, 128, 128)
        x = nn.ReLU(inplace=True)(self.bn2(self.conv2(x)))  # (batch_size, 128, 64, 64)
        x = self.resblock2(x)  # (batch_size, 512, 64, 64)
        x = nn.ReLU(inplace=True)(self.bn3(self.conv3(x)))  # (batch_size, 256, 32, 32)
        x = self.resblock3(x)  # (batch_size, 1024, 32, 32)
        x = nn.ReLU(inplace=True)(self.bn4(self.conv4(x)))  # (batch_size, 512, 16, 16)
        x = self.resblock4(x)  # (batch_size, 2048, 16, 16)

        # decoder
        x = nn.ReLU(inplace=True)(self.upbn1(self.upconv1(x)))  # (batch_size, 1024, 32, 32)
        x = nn.ReLU(inplace=True)(self.upbn2(self.upconv2(x)))  # (batch_size, 512, 64, 64)
        x = nn.ReLU(inplace=True)(self.upbn3(self.upconv3(x)))  # (batch_size, 256, 128, 128)
        x = nn.ReLU(inplace=True)(self.upbn4(self.upconv4(x)))  # (batch_size, 128, 256, 256)
        x = nn.ReLU(inplace=True)(self.upbn5(self.upconv5(x)))  # (batch_size, 64, 512, 512)
        x = self.final_conv(x)  # (batch_size, 3, 512, 512)
        print('generator output shape:', x.shape)
        return x


class ResnetGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, resnet_type='resnet18'):
        super(ResnetGenerator, self).__init__()
        self.resnet = None
        if resnet_type == 'resnet50':
            self.resnet = Resnet50(in_channels=in_channels, out_channels=out_channels)

        elif resnet_type == 'resnet18':
            self.resnet = ResNet18(in_channels=in_channels, out_channels=out_channels)

        elif resnet_type == 'resnet34':
            self.resnet = ResNet34(in_channels=in_channels, out_channels=out_channels)

        elif resnet_type == 'resnet101':
            self.resnet = ResNet101(in_channels=in_channels, out_channels=out_channels)

        elif resnet_type == 'resnet152':
            self.resnet = ResNet152(in_channels=in_channels, out_channels=out_channels)

        else:
            self.resnet = Resnet50(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):

        x = self.resnet(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Discriminator, self).__init__()

        # input => deconvolution => flatten => fully connected => probabilities(?)(softmax?, sigmoid?)
        # input => (3, 512, 512)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1)  # 64 * 256 * 256
        self.bn1 = bb.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # 128 * 128 * 128
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)  # 256 * 64 * 64
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)  # 512 * 32 * 32
        self.bn4 = nn.BatchNorm2d(num_features=512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)  # 1024 * 16 * 16
        self.bn5 = nn.BatchNorm2d(num_features=1024)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=1)  # 1024 * 8 * 8
        
        self.out_patch = nn.Conv2d(in_channels=1024, out_channels=out_channels, kernel_size=1)  # 1 * 8 * 8

    def forward(self, x):
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn1(self.conv1(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn2(self.conv2(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn3(self.conv3(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn4(self.conv4(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn5(self.conv5(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.conv6(x))
        # x = x.view(x.size(0), -1)  # (batch_size, 1024 * 8 * 8)
        # x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.fc1(x))
        # x = nn.Sigmoid()(self.fc2(x))  # play part in two split value
        # x = x.squeeze(1)  # (batch_size)
        x = self.out_patch(x)
        print('discriminator output:', x)

        return x


