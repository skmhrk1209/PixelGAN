import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self, linear_params):

        super().__init__()

        self.module_dict = nn.ModuleDict(dict(
            linear_blocks=nn.ModuleList([
                nn.Sequential(
                    nn.Linear(**linear_param),
                    nn.ReLU()
                ) for linear_param in linear_params[:-1]
            ]),
            linear_block=nn.Sequential(
                nn.Linear(**linear_params[-1]),
                nn.Tanh()
            )
        ))

    def forward(self, inputs):

        for i, linear_block in enumerate(self.module_dict.linear_blocks):
            inputs = linear_block(inputs)
            if i and i % 2 == 0:
                inputs = inputs + shortcut
                shortcut = inputs

        inputs = self.module_dict.linear_block(inputs)

        return inputs


class Discriminator(nn.Module):

    def __init__(self, conv_params, linear_param):

        super().__init__()

        self.module_dict = nn.ModuleDict(dict(
            conv_blocks=nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(**conv_param),
                    nn.BatchNorm2d(conv_param.out_channels),
                    nn.ReLU()
                ) for conv_param in conv_params
            ]),
            linear_block=nn.Linear(**linear_param)
        ))

    def forward(self, inputs, labels):

        for conv_block in self.module_dict.conv_blocks:
            inputs = conv_block(inputs)

        inputs = torch.mean(inputs, dim=(2, 3))
        inputs = self.module_dict.linear_block(inputs)

        return inputs
