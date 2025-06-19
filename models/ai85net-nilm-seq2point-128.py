###################################################################################################
#
# Copyright (C) 2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Seq2Point network for MAX7800X
"""
import torch
from torch import nn

import torch.nn.functional as F

import ai8x


class AI85NILMSeq2Point128(nn.Module):
    """
    Small size UNet model
    """
    def __init__(
            self,
            num_classes=5,
            num_channels=1,
            dimensions=(100, 1),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        dropout = 0.25

        self.dropout = nn.Dropout(dropout)

        self.conv1_1 = ai8x.FusedConv1dBNReLU(num_channels, 128, 1, stride=1, padding=0,
                bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv1_2 = ai8x.FusedConv1dBNReLU(128, 64, 3, stride=1, padding=1,
                bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv1_3 = ai8x.FusedConv1dBNReLU(64, 128, 3, stride=1, padding=1,
                bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv2_1 = ai8x.FusedMaxPoolConv1dBNReLU(128, 128, 3, stride=1, padding=1,
                bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv2_2 = ai8x.FusedConv1dBNReLU(128, 64, 1, stride=1, padding=0,
                bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv2_3 = ai8x.FusedConv1dBNReLU(64, 128, 1, stride=1, padding=0,
                bias=bias, batchnorm='NoAffine', **kwargs)
        
        self.conv3_1 = ai8x.FusedMaxPoolConv1dBNReLU(128, 128, 3, stride=1, padding=1,
                bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv3_2 = ai8x.FusedConv1dBNReLU(128, 64, 5, stride=1, padding=2,
                bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv4_1 = ai8x.FusedMaxPoolConv1dBNReLU(64, 128, 5, stride=1, padding=2,
                bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv4_2 = ai8x.FusedConv1dBNReLU(128, 128, 1, stride=1, padding=0,
                bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv5_1 = ai8x.FusedMaxPoolConv1dBNReLU(128, 128, 5, stride=1, padding=2,
                bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv5_2 = ai8x.FusedConv1dBNReLU(128, 64, 3, stride=1, padding=1,
                bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv6_1 = ai8x.FusedMaxPoolConv1dBNReLU(64, 64, 5, stride=1, padding=2,
                bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv6_2 = ai8x.FusedConv1dBNReLU(64, 64, 1, stride=1, padding=0,
                bias=bias, batchnorm='NoAffine', **kwargs)

        self.mlp1 = ai8x.FusedLinearReLU(256, 256, bias=bias, **kwargs)

        self.fc_state = ai8x.Linear(256, num_classes*2, bias=bias, activation="Abs", **kwargs)
        self.fc_power = ai8x.Linear(256, num_classes*5, bias=bias, **kwargs)

        self.initWeights("kaiming")

    def forward(self, x):
        x = self.conv1_1(x)       # 128
        x = self.conv1_2(x)       # 128
        x = self.conv1_3(x)       # 128
        x = self.conv2_1(x)       # 64
        x = self.conv2_2(x)       # 64
        x = self.conv2_3(x)       # 64
        x = self.conv3_1(x)       # 32
        x = self.conv3_2(x)       # 32
        x = self.conv4_1(x)       # 16
        x = self.conv4_2(x)       # 16
        x = self.conv5_1(x)       # 8
        x = self.conv5_2(x)       # 8
        x = self.conv6_1(x)       # 4
        x = self.conv6_2(x)       # 4
        x = self.dropout(x)
        x = x.view(x.size(0), -1) # 256
        x = self.mlp1(x)
        x1 = self.fc_state(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.fc_power(x)
        x2 = x2.view(x2.size(0), -1)
        return torch.cat([x1, x2], dim=1)

    def initWeights(self, weight_init="kaiming"):
        """
        Auto Encoder Weight Initialization
        """
        weight_init = weight_init.lower()
        assert weight_init in ('kaiming', 'xavier', 'glorot')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if weight_init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif weight_init in ('glorot', 'xavier'):
                    nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, nn.ConvTranspose2d):
                if weight_init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif weight_init in ('glorot', 'xavier'):
                    nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, nn.Linear):
                if weight_init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif weight_init in ('glorot', 'xavier'):
                    nn.init.xavier_uniform_(m.weight)


def ai85netnilmseq2point128(pretrained=False, **kwargs):
    """
    Constructs a AI85KWS20Net model.
    """
    assert not pretrained
    return AI85NILMSeq2Point128(**kwargs)


models = [
    {
        'name': 'ai85netnilmseq2point128',
        'min_input': 1,
        'dim': 1,
    },
]