import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self, linear_params):

        super().__init__()

        self.module_dict = nn.ModuleDict(dict(
            first_linear_block=nn.Sequential(
                nn.Linear(**linear_params[0]),
                nn.Tanh()
            ),
            linear_blocks=nn.ModuleList([
                nn.Sequential(
                    nn.Linear(**linear_param),
                    nn.Tanh() if i % 4 == 0 else nn.ReLU()
                ) for i, linear_param in enumerate(linear_params[1:-1])
            ]),
            last_linear_block=nn.Sequential(
                nn.Linear(**linear_params[-1]),
                nn.Sigmoid()
            )
        ))

    def forward(self, inputs):

        inputs = self.module_dict.first_linear_block(inputs)

        shortcut = inputs
        for i, linear_block in enumerate(self.module_dict.linear_blocks):
            inputs = linear_block(inputs)
            if i and i % 4 == 0:
                inputs = inputs + shortcut
                shortcut = inputs

        inputs = self.module_dict.last_linear_block(inputs)

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
