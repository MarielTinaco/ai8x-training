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


class AI85NILMRSTP(nn.Module):
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
        ## INIT ##
        self.conv1 = ai8x.FusedConv1dBNReLU(num_channels, 32, 1, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)

        ## 1ST ##
        self.conv2 = ai8x.FusedConv1dBNReLU(32, 32, 1, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)

        self.conv3 = ai8x.FusedConv1dBNReLU(32, 32, 1, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)
        self.resid_1 = ai8x.Add()

        ## 2ND ##
        self.conv4 = ai8x.FusedConv1dBNReLU(32, 48, 3, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)
        # residual max pool 2
        self.conv5_max = ai8x.FusedMaxPoolConv1dBNReLU(48, 48, 1, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)
        self.conv6 = ai8x.FusedConv1dBNReLU(48, 48, 1, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)
        self.resid_2 = ai8x.Add()

        ## 3RD ##
        self.conv7 = ai8x.FusedConv1dBNReLU(48, 64, 1, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)
        # residual max pool 3
        self.conv8_max = ai8x.FusedMaxPoolConv1dBNReLU(64, 64, 1, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)
        self.conv9 = ai8x.FusedConv1dBNReLU(64, 64, 1, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)
        self.resid_3 = ai8x.Add()

        ## EXTRA ##
        self.conv10 = ai8x.FusedAvgPoolConv1dBNReLU(64, 92, 4, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)

        self.conv11 = ai8x.FusedConv1dBNReLU(92, 128, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        
        self.conv12 = ai8x.FusedConv1dBNReLU(128, 32, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        
        self.lin = ai8x.FusedLinearReLU(352, 128, bias=bias, **kwargs)

        ## MLP ##
        self.mlp1 = ai8x.FusedLinearReLU(128, 256, bias=bias, **kwargs)

        ## LINEAR ##
        self.fc_state = ai8x.Linear(256, num_classes*2, bias=bias, **kwargs)
        self.fc_power = ai8x.Linear(256, num_classes*5, bias=bias, **kwargs)

        self.initWeights("kaiming")

    def forward(self, x):
        ## INIT ##
        x = self.conv1(x)

        ## 1ST ##
        x_res = self.conv2(x)
        x = self.conv3(x_res)
        x = self.resid_1(x, x_res)

        ## 2ND ##
        x = self.conv4(x)
        x_res = self.conv5_max(x)
        x = self.conv6(x_res)
        x = self.resid_2(x, x_res)
        
        ## 3RD ##
        x = self.conv7(x)
        x_res = self.conv8_max(x)
        x = self.conv9(x_res)
        x = self.resid_3(x, x_res)

        ## EXTRA ##
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
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

def ai85nilmresidual(pretrained=False, **kwargs):
    """
    Constructs a AI85KWS20Net model.
    """
    assert not pretrained
    return AI85NILMRSTP(**kwargs)


models = [
    {
        'name': 'ai85nilmresidual',
        'min_input': 1,
        'dim': 1,
    },
]