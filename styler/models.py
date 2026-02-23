from typing import List

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_dropout: bool, dropout_p: float) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout and dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class StyleGenerator(nn.Module):
    """
    ResNet-based generator (CycleGAN-like)
    """
    def __init__(self, n_residual_blocks: int = 9, use_dropout: bool = False, dropout_p: float = 0.5) -> None:
        super().__init__()
        c7s1_64 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        down = [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]

        # Residual blocks
        res = [ResidualBlock(256, use_dropout=use_dropout, dropout_p=dropout_p) for _ in range(n_residual_blocks)]

        # Upsampling
        up = [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        out = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*(c7s1_64 + down + res + up + out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """
    PatchGAN discriminator
    """
    def __init__(self) -> None:
        super().__init__()

        def block(in_c: int, out_c: int, norm: bool = True) -> nn.Sequential:
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(3, 64, norm=False),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)
