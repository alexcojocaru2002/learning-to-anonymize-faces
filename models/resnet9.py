from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_dropout=False, padding_type='reflect'):
        super().__init__()
        pad = {
            'reflect': nn.ReflectionPad2d,
            'replicate': nn.ReplicationPad2d,
            'zero': lambda p: nn.ZeroPad2d(p)
        }[padding_type]

        layers = [
            pad(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            layers.append(nn.Dropout(0.5))

        layers += [
            pad(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels)
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class ResNet(nn.Module):
    # ngf = number of filters in first conv layer
    # basically output channels la fiecare resnet block
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9, use_dropout=False, padding_type='reflect'):
        super().__init__()
        assert n_blocks >= 0, "Number of residual blocks must be non-negative"

        Pad = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d

        self.initial = nn.Sequential(
            Pad(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        # Downsampling layers
        down_layers = []
        in_ch = ngf
        for _ in range(2):
            out_ch = in_ch * 2
            down_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch
        self.down = nn.Sequential(*down_layers)

        # Residual blocks
        # should check if we need to use bias here
        res_blocks = [ResidualBlock(in_ch, use_dropout, padding_type) for _ in range(n_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Upsampling layers
        up_layers = []
        for _ in range(2):
            out_ch = in_ch // 2
            up_layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch
        self.up = nn.Sequential(*up_layers)

        # Final layer
        self.final = nn.Sequential(
            Pad(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down(x)
        x = self.res_blocks(x)
        x = self.up(x)
        return self.final(x)