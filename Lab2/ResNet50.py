from torch import nn
import torch

from torchinfo import summary

# print("Please define your ResNet50 in this file.")

class Bottleneck_Block(nn.Module):
    expansion = 4

    def __init__(self, input_shape, hidden_shape, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=input_shape,
                               out_channels=hidden_shape,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_shape)
        self.conv2 = nn.Conv2d(in_channels=hidden_shape,
                               out_channels=hidden_shape,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_shape)
        self.conv3 = nn.Conv2d(in_channels=hidden_shape, 
                               out_channels=hidden_shape * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_shape * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or input_shape != hidden_shape * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=input_shape,
                          out_channels=hidden_shape * self.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(hidden_shape * self.expansion)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += residual
        x = torch.relu(x)
        return x



class ResNet50(nn.Module):
    def __init__(self, input_shape: int,
                #  hidden_units: list,
                 output_shape: int) -> None:
        super().__init__()
        self.in_channels = 64

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=64,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),   # size: 64*112*112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1) # size: 64*56*56
        )
        
        self.conv_block_2 = self._make_layer(hidden_size=64, 
                                             blocks=3, 
                                             stride=1)
        
        self.conv_block_3 = self._make_layer(hidden_size=128, 
                                             blocks=4, 
                                             stride=2)
        
        self.conv_block_4 = self._make_layer(hidden_size=256, 
                                             blocks=6, 
                                             stride=2)
        
        self.conv_block_5 = self._make_layer(hidden_size=512, 
                                             blocks=3, 
                                             stride=2)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512 * Bottleneck_Block.expansion, 
                      out_features=output_shape)
        )


    def _make_layer(self, hidden_size, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck_Block(input_shape=self.in_channels, 
                                           hidden_shape=hidden_size, 
                                           stride=stride))
            self.in_channels = hidden_size * Bottleneck_Block.expansion
        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.classifier(x)
        return x

# model = ResNet50(input_shape=3,
#                  output_shape=100)
# print(model)
model_resnet = ResNet50(input_shape=3,
                      output_shape=100)
summary(model_resnet, input_size=[1, 3, 224, 224])