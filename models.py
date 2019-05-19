import torch
from torch import nn


class VAE(nn.Module):

    def __init__(self, conv_params, linear_params):

        super().__init__()

        self.module_dict = nn.ModuleDict(dict(
            conv_blocks=nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(**conv_param),
                    nn.ReLU()
                ) for conv_param in conv_params
            ]),
            linear_blocks=nn.ModuleList([
                nn.Sequential(
                    nn.Linear(**linear_param),
                    nn.ReLU()
                ) for linear_param in linear_params[:-1]
            ]),
            linear_block=nn.Sequential(
                nn.Linear(**linear_params[-1])
            )
        ))

    def forward(self, inputs):

        for conv_block in self.module_dict.conv_blocks:
            inputs = conv_block(inputs)

        inputs = inputs.reshape(inputs.size(0), -1)

        for linear_block in self.module_dict.linear_blocks:
            inputs = linear_block(inputs)

        inputs = self.module_dict.linear_block(inputs)
        means, logvars = torch.chunk(inputs, 2, dim=1)

        latents = torch.randn_like(means) * torch.exp(0.5 * logvars) + means
        kl_divergences = -0.5 * torch.sum(1 + logvars - torch.pow(means, 2) - torch.exp(logvars), dim=1)

        return latents, kl_divergences


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
                    nn.Sequential(
                        nn.Linear(**linear_param),
                        nn.ReLU()
                    ),
                    nn.Sequential(
                        nn.Linear(**linear_param),
                        nn.ReLU()
                    ),
                    nn.Sequential(
                        nn.Linear(**linear_param),
                        nn.ReLU()
                    ),
                    nn.Sequential(
                        nn.Linear(**linear_param),
                        nn.Tanh()
                    )
                ) for linear_param in linear_params[1:-1]
            ]),
            last_linear_block=nn.Sequential(
                nn.Linear(**linear_params[-1]),
                nn.Sigmoid()
            )
        ))

    def forward(self, inputs):

        inputs = self.module_dict.first_linear_block(inputs)

        for i, linear_block in enumerate(self.module_dict.linear_blocks):
            inputs += linear_block(inputs)

        inputs = self.module_dict.last_linear_block(inputs)

        return inputs


class Discriminator(nn.Module):

    def __init__(self, conv_params, linear_params):

        super().__init__()

        self.module_dict = nn.ModuleDict(dict(
            conv_blocks=nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(**conv_param),
                    nn.ReLU()
                ) for conv_param in conv_params
            ]),
            linear_blocks=nn.ModuleList([
                nn.Sequential(
                    nn.Linear(**linear_param),
                    nn.ReLU()
                ) for linear_param in linear_params[:-1]
            ]),
            linear_block=nn.Sequential(
                nn.Linear(**linear_params[-1])
            )
        ))

    def forward(self, inputs):

        for conv_block in self.module_dict.conv_blocks:
            inputs = conv_block(inputs)

        inputs = inputs.reshape(inputs.size(0), -1)

        for linear_block in self.module_dict.linear_blocks:
            inputs = linear_block(inputs)

        inputs = self.module_dict.linear_block(inputs)

        return inputs
