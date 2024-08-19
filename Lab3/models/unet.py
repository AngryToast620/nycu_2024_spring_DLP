# Implement your UNet model here
import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, input_shape,
                 output_shape):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=output_shape,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(output_shape),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_shape,
                      out_channels=output_shape,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(output_shape),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)
    


class UNet(nn.Module):
    def __init__(self, input_shape,
                 output_shape):
        super().__init__()

        self.encoder = nn.Sequential(
            DoubleConv(input_shape=input_shape,
                       output_shape=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(input_shape=64,
                       output_shape=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(input_shape=128,
                       output_shape=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(input_shape=256,
                       output_shape=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(input_shape=512,
                       output_shape=1024)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,
                               out_channels=512, 
                               kernel_size=2,
                               stride=2),
            DoubleConv(input_shape=1024,
                       output_shape=512),
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=2,
                               stride=2),
            DoubleConv(input_shape=512,
                       output_shape=256),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=2,
                               stride=2),
            DoubleConv(input_shape=256,
                       output_shape=128),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=2,
                               stride=2),
            DoubleConv(input_shape=128,
                       output_shape=64),
            nn.Conv2d(in_channels=64,
                      out_channels=output_shape,
                      kernel_size=1)
        )

    def forward(self, x):
        encodes = []
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)
            if i % 2 == 0:
                encodes.append(x)
        encodes.pop()
        decodes = encodes.pop()
        for j, decoder in enumerate(self.decoder):
            if j % 2 == 1:
                x = torch.cat([x, decodes], dim=1)
                if encodes:
                    decodes = encodes.pop()
            x = decoder(x)
        return x
    
# a = UNet(3, 1)
# for i in a.encoder:
#     print('----------')
#     print(i)