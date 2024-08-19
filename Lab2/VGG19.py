from torch import nn
import torch
from torchinfo import summary

# print("Please define your VGG19 in this file.")

class VGG19(nn.Module):
    def __init__(self, input_shape: int,
                #  hidden_units: list,
                 output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = self._make_layer(input_shape=input_shape,
                                             hidden_size=64,
                                             blocks=2)
        self.conv_block_2 = self._make_layer(input_shape=64,
                                             hidden_size=128,
                                             blocks=2)
        self.conv_block_3 = self._make_layer(input_shape=128,
                                             hidden_size=256,
                                             blocks=4)
        self.conv_block_4 = self._make_layer(input_shape=256,
                                             hidden_size=512,
                                             blocks=4)
        self.conv_block_5 = self._make_layer(input_shape=512,
                                             hidden_size=512,
                                             blocks=4)

        # self.conv_block_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=input_shape,
        #               out_channels=64,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64,
        #               out_channels=64,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2,
        #                  stride=2)
        # )

        # self.conv_block_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64,
        #               out_channels=128,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=128,
        #               out_channels=128,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2,
        #                  stride=2)
        # )

        # self.conv_block_3 = nn.Sequential(
        #     nn.Conv2d(in_channels=128,
        #               out_channels=256,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=256,
        #               out_channels=256,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=256,
        #               out_channels=256,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=256,
        #               out_channels=256,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2,
        #                  stride=2)
        # )

        # self.conv_block_4 = nn.Sequential(
        #     nn.Conv2d(in_channels=256,
        #               out_channels=512,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=512,
        #               out_channels=512,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=512,
        #               out_channels=512,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=512,
        #               out_channels=512,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2,
        #                  stride=2)
        # )

        # self.conv_block_5 = nn.Sequential(
        #     nn.Conv2d(in_channels=512,
        #               out_channels=512,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=512,
        #               out_channels=512,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=512,
        #               out_channels=512,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=512,
        #               out_channels=512,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2,
        #                  stride=2)
        # )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512*7*7,
                      out_features=4096),
            nn.Softmax(dim=0),
            nn.Linear(in_features=4096,
                      out_features=4096),
            nn.Softmax(dim=0),
            nn.Linear(in_features=4096,
                      out_features=1000),
            nn.Softmax(dim=0),
            nn.Linear(in_features=1000,
                      out_features=output_shape)
        )

    def _make_layer(self, input_shape, hidden_size, blocks):
        layers = []
        for i in range(blocks):
            if(i == 0):
                layers.append(nn.Conv2d(in_channels=input_shape,
                                        out_channels=hidden_size,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1))
            else:
                layers.append(nn.Conv2d(in_channels=hidden_size,
                                        out_channels=hidden_size,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2,
                                   stride=2))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        # x = torch.flatten(x)
        x = self.classifier(x)
        return x
    
model_resnet = VGG19(input_shape=3,
                      output_shape=100)
summary(model_resnet, input_size=[1, 3, 224, 224])