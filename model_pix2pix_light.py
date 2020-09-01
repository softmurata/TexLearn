# light => 128 * 128
# middle => 256 * 256
# heavy => 512 * 512

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
    
    def __init__(self, in_channels, out_channels, mode='nearest'):
        super(UnetGenerator, self).__init__()
        self.mode = mode
        # input => (128, 128, 3)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1)  # 64 * 64 * 64
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)  # 128 * 32 * 32
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)  # 256 * 16 * 16
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)  # 512 * 8 * 8
        self.bn4 = nn.BatchNorm2d(num_features=512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)  # 512 * 4 * 4
        self.bn5 = nn.BatchNorm2d(num_features=512)
        
        # upsample
        if self.mode:
            self.upsample1 = nn.Upsample(scale_factor=2, mode=self.mode)  # 512 * 8 * 8
            self.deconv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)  # 256 * 8 * 8
            self.debn1 = nn.BatchNorm2d(num_features=256)
            
            self.upsample2 = nn.Upsample(scale_factor=2, mode=self.mode)  # 256 * 16 * 16
            self.deconv2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)  # 128 * 16 * 16
            self.debn2 = nn.BatchNorm2d(num_features=128)
            
            self.upsample3 = nn.Upsample(scale_factor=2, mode=self.mode)  # 128 * 32 * 32
            self.deconv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)  # 64 * 32 * 32
            self.debn3 = nn.BatchNorm2d(num_features=64)
            
            self.upsample4 = nn.Upsample(scale_factor=2, mode=self.mode)  # 64 * 64 * 64
            self.deconv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)  # 64 * 64 * 64
            self.debn4 = nn.BatchNorm2d(num_features=64)
            
            self.upsample5 = nn.Upsample(scale_factor=2, mode=self.mode)  # 64 * 128 * 128
            self.deconv5 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1)  # 3 * 128 * 128
        
        else:
            self.upconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0)  # 512 * 8 * 8
            self.convblock1 = ConvBlock(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1) # 256 * 8 * 8
            
            self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0)  # 256 * 16 * 16
            self.convblock2 = ConvBlock(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1) # 128 * 16 * 16
            
            self.upconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0)  # 128 * 32 * 32
            self.convblock3 = ConvBlock(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)  # 64 * 32 * 32
            
            self.upconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)  # 64 * 64 * 64
            self.convblock4 = ConvBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)  # 64 * 64 * 64
            
            self.upconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=2, stride=2, padding=0)  # 3 * 128 * 128
            
            
            
    def forward(self, x):
        # x => 3 * 128 * 128
        print('input:', x.shape)
        # encoder
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn1(self.conv1(x)))  # (batch_size, 64, 64, 64)
        x1 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn2(self.conv2(x)))  # (batch_size, 128, 32, 32)
        x2 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn3(self.conv3(x)))  # (batch_size, 256, 16, 16)
        x3 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn4(self.conv4(x)))  # (batch_size, 512, 8, 8)
        x4 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn5(self.conv5(x)))  # (batch_size, 512, 4, 4)
        
        print('encoder shape:', x.shape)
        
        # decoder
        if self.mode:
            x = self.upsample1(x)  # (batch_size, 512, 8, 8)
            x = torch.cat([x4, x], dim=1)  # (batch_size, 1024, 8, 8)
            x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.debn1(self.deconv1(x)))  # (batch_size, 256, 8, 8)
            
            x = self.upsample2(x)  # (batch_size, 256, 16, 16)
            x = torch.cat([x3, x], dim=1)  # (batch_size, 512, 16, 16)
            x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.debn2(self.deconv2(x)))  # (batch_size, 128, 16, 16)
            
            x = self.upsample3(x)  # (batch_size, 128, 32, 32)
            x = torch.cat([x2, x], dim=1)  # (batch_size, 256, 32, 32)
            x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.debn3(self.deconv3(x)))  # (batch_size, 64, 32, 32)
            
            x = self.upsample4(x)  # (batch_size, 64, 64, 64)
            x = torch.cat([x1, x], dim=1)  # (batch_size, 128, 64, 64)
            x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.debn4(self.deconv4(x)))  # (batch_size, 64, 64, 64)
            
            x = self.upsample5(x)  # (batch_size, 64, 128, 128)
            x = nn.Tanh()(self.deconv5(x))  # (batch_size, 3, 128, 128)
            
        else:
            x = self.upconv1(x)  # (batch_size, 512, 8, 8)
            x = torch.cat([x4, x], dim=1)  # (batch_size, 1024, 8, 8)
            x = self.convblock1(x)  # (batch_size, 256, 8, 8)
            
            x = self.upconv2(x)  # (batch_size, 256, 16, 16)
            x = torch.cat([x3, x], dim=1)  # (batch_size, 512, 16, 16)
            x = self.convblock2(x)  # (batch_size, 128, 16, 16)
            
            x = self.upconv3(x)  # (batch_size, 128, 32, 32)
            x = torch.cat([x2, x], dim=1)  # (batch_size, 256, 32, 32)
            x = self.convblock1(x)  # (batch_size, 64, 32, 32)
            
            x = self.upconv4(x)  # (batch_size, 64, 64, 64)
            x = torch.cat([x1, x], dim=1)  # (batch_size, 128, 64, 64)
            x = self.convblock1(x)  # (batch_size, 64, 64, 64)
            
            x = nn.Tanh()(self.upconv5(x))  # (batch_size, 3, 128, 128)
        
        print('decoder shape:', x.shape)
            
        return x


class ResnetGenerator(nn.Module):
    
    def __init__(self, in_channels, out_channels, resnet_type):
        super(ResnetGenerator, self).__init__()
        
    def forward(self, x):
        
        return x
    
    

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Discriminator, self).__init__()

        # input => deconvolution => flatten => fully connected => probabilities(?)(softmax?, sigmoid?)
        # input => (3, 128, 128)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1)  # 64 * 64 * 64
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # 128 * 32 * 32
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)  # 256 * 16 * 16
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)  # 512 * 8 * 8
        self.bn4 = nn.BatchNorm2d(num_features=512)
        self.out_patch = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=1)  # 1 * 8 * 8

    def forward(self, x):
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn1(self.conv1(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn2(self.conv2(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn3(self.conv3(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn4(self.conv4(x)))
        x = self.out_patch(x)  # (batch_size, 1, 8, 8)
        print('discriminator output:', x)

        return x

