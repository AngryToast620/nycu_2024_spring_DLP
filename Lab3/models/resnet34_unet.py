import torch
import torch.nn as nn


class Residual_Block(nn.Module):
    expansion = 1

    def __init__(self, input_shape,
                 output_shape,
                 stride=1):
        super().__init__()

        self.basic = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=output_shape,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(output_shape),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_shape,
                      out_channels=output_shape,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(output_shape),
        )
        self.shortcut = None
        if stride != 1 or input_shape != self.expansion * output_shape:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=input_shape,
                          out_channels=self.expansion * output_shape,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * output_shape)
            )

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        x = self.basic(x)
        x += residual
        x = torch.relu(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encodes = []
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, output_shape,
                    blocks,
                    stride=1):
        layers = []
        layers.append(Residual_Block(self.in_channels, output_shape, stride))
        self.in_channels = output_shape * Residual_Block.expansion
        for _ in range(1, blocks):
            layers.append(Residual_Block(self.in_channels, output_shape))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        self.encodes.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        self.encodes.append(x)
        x = self.layer2(x)
        self.encodes.append(x)
        x = self.layer3(x)
        self.encodes.append(x)
        x = self.layer4(x)
        self.encodes.append(x)
        return x


class UNetDecoderBlock(nn.Module):
    def __init__(self, input_shape,
                 output_shape):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=output_shape,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(output_shape),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_shape,
                      out_channels=output_shape,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(output_shape),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class ResNet_UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = ResNetEncoder()
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        self.upconv1 = nn.ConvTranspose2d(in_channels=1024,
                                          out_channels=512,
                                          kernel_size=2,
                                          stride=2)
        self.upconv2 = nn.ConvTranspose2d(in_channels=512,
                                          out_channels=256,
                                          kernel_size=2,
                                          stride=2)
        self.upconv3 = nn.ConvTranspose2d(in_channels=256,
                                          out_channels=128,
                                          kernel_size=2,
                                          stride=2)
        self.upconv4 = nn.ConvTranspose2d(in_channels=128,
                                          out_channels=64,
                                          kernel_size=2,
                                          stride=2)
        self.upconv5 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=64,
                                          kernel_size=2,
                                          stride=2)
        self.decoder_block1 = UNetDecoderBlock(input_shape=1024,
                                               output_shape=512)
        self.decoder_block2 = UNetDecoderBlock(input_shape=512,
                                               output_shape=256)
        self.decoder_block3 = UNetDecoderBlock(input_shape=256,
                                               output_shape=128)
        self.decoder_block4 = UNetDecoderBlock(input_shape=128,
                                               output_shape=64)
        self.decoder_block5 = UNetDecoderBlock(input_shape=128,
                                               output_shape=64)
        self.last_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.bridge(x)

        x = self.upconv1(x)
        decodes = self.resnet.encodes.pop()
        x = torch.cat([x, decodes], dim=1)
        x = self.decoder_block1(x)

        x = self.upconv2(x)
        decodes = self.resnet.encodes.pop()
        x = torch.cat([x, decodes], dim=1)
        x = self.decoder_block2(x)

        x = self.upconv3(x)
        decodes = self.resnet.encodes.pop()
        x = torch.cat([x, decodes], dim=1)
        x = self.decoder_block3(x)

        x = self.upconv4(x)
        decodes = self.resnet.encodes.pop()
        x = torch.cat([x, decodes], dim=1)
        x = self.decoder_block4(x)

        x = self.upconv5(x)
        decodes = self.resnet.encodes.pop()
        x = torch.cat([x, decodes], dim=1)
        x = self.decoder_block5(x)

        x = self.last_conv(x)
        return x
