###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
SimpleNet_v1 network with added residual layers for AI85.
Simplified version of the network proposed in [1].

[1] HasanPour, Seyyed Hossein, et al. "Lets keep it simple, using simple architectures to
    outperform deeper and more complex architectures." arXiv preprint arXiv:1608.06037 (2016).
"""
from torch import nn

import torch
import ai8x


class AI85ResidualSimpleNet(nn.Module):
    """
    Residual SimpleNet v1 Model
    """
    def __init__(
            self,
            num_classes=5,
            num_channels=32,
            dimensions=(32, 36),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()
        dropout = 0.25

        self.dropout = nn.Dropout(dropout)
        ## INIT ##
        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 32, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)

        ## 1ST ##
        self.conv2 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)

        self.conv3 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        self.resid_1 = ai8x.Add()

        ## 2ND ##
        self.conv4 = ai8x.FusedConv2dBNReLU(32, 48, 3, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)
        # residual max pool 2
        self.conv5_max = ai8x.FusedMaxPoolConv2dBNReLU(48, 48, 1, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)
        self.conv6 = ai8x.FusedConv2dBNReLU(48, 48, 1, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)
        self.resid_2 = ai8x.Add()

        ## 3RD ##
        self.conv7 = ai8x.FusedConv2dBNReLU(48, 64, 1, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)
        # residual max pool 3
        self.conv8_max = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 1, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)
        self.conv9 = ai8x.FusedConv2dBNReLU(64, 64, 1, stride=1, padding=0,
                bias=bias, batchnorm='Affine', **kwargs)
        self.resid_3 = ai8x.Add()

        ## 4TH ##
        self.conv10 = ai8x.FusedConv2dBNReLU(64, 92, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        self.conv11_max = ai8x.FusedMaxPoolConv2dBNReLU(92, 64, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        self.conv12 = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        self.resid_4 = ai8x.Add()

        # ## EXTRA ##
        # self.conv13 = ai8x.FusedAvgPoolConv2dBNReLU(92, 128, 3, stride=1, padding=0,
        #         bias=bias, batchnorm='Affine', **kwargs)

        # self.conv14 = ai8x.FusedConv2dBNReLU(128, 96, 3, stride=1, padding=1,
        #         bias=bias, batchnorm='Affine', **kwargs)
        
        # self.conv15 = ai8x.FusedConv2dBNReLU(96, 64, 3, stride=1, padding=1,
        #         bias=bias, batchnorm='Affine', **kwargs)
        
        self.lin = ai8x.FusedLinearReLU(768, 128, bias=bias, **kwargs)

        self.mlp1 = ai8x.FusedLinearReLU(128, 256, bias=bias, **kwargs)

        self.fc_state = ai8x.Linear(256, num_classes*2, bias=bias, **kwargs)
        self.fc_power = ai8x.Linear(256, num_classes*5, bias=bias, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        #                               # Conv: [(W−K+2P)/S]+1
                                        # Pool: [(W-K)/S] + 1
        ## INIT ##                      # 1x100
        x = self.conv1(x)               # 32x100

        ## 1ST ##
        x_res = self.conv2(x)           # 32x100
        x = self.conv3(x_res)           # 32x100
        x = self.resid_1(x, x_res)      # 32x100

        ## 2ND ##
        x = self.conv4(x)               # 48x98
        x_res = self.conv5_max(x)       # 48x49 (Conv+Pool)
        x = self.conv6(x_res)           # 48x49
        x = self.resid_2(x, x_res)      # 48x49
        
        ## 3RD ##
        x = self.conv7(x)               # 64x49
        x_res = self.conv8_max(x)       # 64x24 (Conv+Pool+rounddown)
        x = self.conv9(x_res)           # 64x24
        x = self.resid_3(x, x_res)      # 64x24

        ## 4TH ##
        x = self.conv10(x)              # 92x24
        x_res = self.conv11_max(x)      # 92x12 (Conv+Pool+rounddown)
        x = self.conv12(x_res)          # 92x12
        x = self.resid_4(x, x_res)      # 92x12

        ## EXTRA ##
        # x = self.conv13(x)              # 128x5
        # x = self.conv14(x)              # 96x5
        # x = self.conv15(x)              # 64x5
        x = self.dropout(x)             # 64x5
        x = x.view(x.size(0), -1)       # 320
        x = self.lin(x)
        x = self.mlp1(x)
        x1 = self.fc_state(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.fc_power(x)
        x2 = x2.view(x2.size(0), -1)
        return torch.cat([x1, x2], dim=1)


def ai85nilmslidingwindowressimplenet(pretrained=False, **kwargs):
    """
    Constructs a Residual SimpleNet v1 model.
    """
    assert not pretrained
    return AI85ResidualSimpleNet(**kwargs)


models = [
    {
        'name': 'ai85nilmslidingwindowressimplenet',
        'min_input': 1,
        'dim': 2,
    },
]
