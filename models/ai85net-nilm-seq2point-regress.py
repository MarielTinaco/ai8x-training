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


class AI85NILMSeq2PointRegress(nn.Module):
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

        dropout = 0.25

        self.dropout = nn.Dropout(dropout)

        self.conv1 = ai8x.FusedConv1dBNReLU(num_channels, 32, 9, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)

        self.conv2 = ai8x.FusedConv1dBNReLU(32, 32, 8, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)

        self.conv3 = ai8x.FusedMaxPoolConv1dBNReLU(32, 48, 6, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)

        self.conv4 = ai8x.FusedMaxPoolConv1dBNReLU(48, 64, 5, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)

        self.conv5 = ai8x.FusedConv1dBNReLU(64, 64, 5, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)

        self.conv6 = ai8x.FusedAvgPoolConv1dBNReLU(64, 92, 4, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)

        self.conv7 = ai8x.FusedConv1dBNReLU(92, 128, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        
        self.conv8 = ai8x.FusedConv1dBNReLU(128, 32, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)

        self.mlp1 = ai8x.FusedLinearReLU(128, 256, bias=bias, **kwargs)

        self.fc_state = ai8x.Linear(256, num_classes*2, bias=bias, **kwargs)
        self.fc_power = ai8x.Linear(256, num_classes*5, bias=bias, **kwargs)

        self.initWeights("kaiming")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
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

def ai85nilmseq2pointregress(pretrained=False, **kwargs):
    """
    Constructs a AI85KWS20Net model.
    """
    assert not pretrained
    return AI85NILMSeq2PointRegress(**kwargs)


models = [
    {
        'name': 'ai85nilmseq2pointregress',
        'min_input': 1,
        'dim': 1,
    },
]